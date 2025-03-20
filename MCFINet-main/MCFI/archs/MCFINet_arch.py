import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision import ops
from basicsr.utils.registry import ARCH_REGISTRY

def silu(x):
    return x * F.sigmoid(x)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x, z):
        x = x * silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Mamba2(nn.Module):
    def __init__(self, d_model: int,  # model dimension (D)
                 n_layer: int = 24,  # number of Mamba-2 layers in the language model
                 d_state: int = 128,  # state dimension (N)
                 d_conv: int = 4,  # convolution kernel size
                 expand: int = 2,  # expansion factor (E)
                 headdim: int = 64,  # head dimension (P)
                 chunk_size: int = 64,  # matrix partition size (Q)
                 ):
        super().__init__()
        self.n_layer = n_layer
        self.d_state = d_state
        self.headdim = headdim
        # self.chunk_size = torch.tensor(chunk_size, dtype=torch.int32)
        self.chunk_size = chunk_size

        self.d_inner = expand * d_model
        assert self.d_inner % self.headdim == 0, "self.d_inner must be divisible by self.headdim"
        self.nheads = self.d_inner // self.headdim

        d_in_proj = 2 * self.d_inner + 2 * self.d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        conv_dim = self.d_inner + 2 * d_state
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, d_conv, groups=conv_dim, padding=d_conv - 1, )
        self.dt_bias = nn.Parameter(torch.empty(self.nheads, ))
        self.A_log = nn.Parameter(torch.empty(self.nheads, ))
        self.D = nn.Parameter(torch.empty(self.nheads, ))
        self.norm = RMSNorm(self.d_inner, )
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False, )

    def forward(self, u: Tensor):
        A = -torch.exp(self.A_log)  # (nheads,)
        zxbcdt = self.in_proj(u)  # (batch, seqlen, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.d_inner,
                self.d_inner + 2 * self.d_state,
                self.nheads,
            ],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)  # (batch, seqlen, nheads)

        # Pad or truncate xBC seqlen to d_conv
        xBC = silu(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, : u.shape[1], :]
        )  # (batch, seqlen, d_inner + 2 * d_state))
        x, B, C = torch.split(
            xBC, [self.d_inner, self.d_state, self.d_state], dim=-1
        )

        _b, _l, _hp = x.shape
        _h = _hp // self.headdim
        _p = self.headdim
        x = x.reshape(_b, _l, _h, _p)

        y = self.ssd(x * dt.unsqueeze(-1),
                     A * dt,
                     B.unsqueeze(2),
                     C.unsqueeze(2), )

        y = y + x * self.D.unsqueeze(-1)

        _b, _l, _h, _p = y.shape
        y = y.reshape(_b, _l, _h * _p)

        y = self.norm(y, z)
        y = self.out_proj(y)

        return y

    def segsum(self, x: Tensor) -> Tensor:
        T = x.size(-1)
        device = x.device
        x = x[..., None].repeat(1, 1, 1, 1, T)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
        x = x.masked_fill(~mask, 0)
        x_segsum = torch.cumsum(x, dim=-2)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum

    def ssd(self, x, A, B, C):
        chunk_size = self.chunk_size
        # if x.shape[1] % chunk_size == 0:
        #
        x = x.reshape(x.shape[0], x.shape[1] // chunk_size, chunk_size, x.shape[2], x.shape[3], )
        B = B.reshape(B.shape[0], B.shape[1] // chunk_size, chunk_size, B.shape[2], B.shape[3], )
        C = C.reshape(C.shape[0], C.shape[1] // chunk_size, chunk_size, C.shape[2], C.shape[3], )
        A = A.reshape(A.shape[0], A.shape[1] // chunk_size, chunk_size, A.shape[2])
        A = A.permute(0, 3, 1, 2)
        A_cumsum = torch.cumsum(A, dim=-1)

        # 1. Compute the output for each intra-chunk (diagonal blocks)
        L = torch.exp(self.segsum(A))
        Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

        # 2. Compute the state for each intra-chunk
        # (right term of low-rank factorization of off-diagonal blocks; B terms)
        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

        # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
        # (middle term of factorization of off-diag blocks; A terms)

        initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)

        decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))[0]
        new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
        states = new_states[:, :-1]

        # 4. Compute state -> output conversion per chunk
        # (left term of low-rank factorization of off-diagonal blocks; C terms)
        state_decay_out = torch.exp(A_cumsum)
        Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

        # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
        # Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
        Y = Y_diag + Y_off
        Y = Y.reshape(Y.shape[0], Y.shape[1] * Y.shape[2], Y.shape[3], Y.shape[4], )

        return Y
class BiMamba2_2D(nn.Module):
    def __init__(self, cin, cout, d_model, **mamba2_args):
        super().__init__()
        self.fc_in = nn.Linear(cin, d_model, bias=False)  # 调整通道数到 d_model
        self.mamba2_for = Mamba2(d_model, **mamba2_args)  # 正向 Mamba2
        self.mamba2_back = Mamba2(d_model, **mamba2_args)  # 反向 Mamba2
        self.fc_out = nn.Linear(d_model, cout, bias=False)  # 调整通道数到 cout

    def forward(self, x):
        # 输入形状: (batch_size, channels, height, width)
        batch_size, channels, height, width = x.shape
        x = x.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)  # 转换为 (batch_size, seq_len, channels)
        x = self.fc_in(x)  # 调整通道数为 d_model

        # 正向和反向处理
        x1 = self.mamba2_for(x)  # 正向 Mamba2
        x2 = self.mamba2_back(x.flip(1)).flip(1)  # 反向 Mamba2
        x = x1 + x2  # 融合正向和反向结果

        x = self.fc_out(x)  # 调整通道数为 cout
        x = x.reshape(batch_size, height, width, -1)  # 恢复形状
        x = x.permute(0, 3, 1, 2)  # 转换为 (batch_size, channels, height, width)
        return x


class AdaptiveLFE(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)  # 初始特征提取卷积
        self.adaptive_conv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim)  # 使用自适应权重的深度卷积
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)  # 输出通道恢复卷积
        self.act = nn.GELU()
        self.alpha = nn.Parameter(torch.ones((1, hidden_dim, 1, 1)))  # 自适应参数
        self.beta = nn.Parameter(torch.zeros((1, hidden_dim, 1, 1)))  # 自适应参数

    def forward(self, x):
        x = self.conv_0(x)
        x = self.adaptive_conv(x) * self.alpha + self.beta  # 应用自适应权重进行非局部增强
        x = self.act(x)
        x = self.conv_1(x)  # 输出恢复通道
        return x


class ACFB(nn.Module):
    def __init__(self, dim, growth_rate=2.0, p_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        p_dim = int(hidden_dim * p_rate)
        self.conv_0 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.conv_1 = nn.Conv2d(p_dim, p_dim, 3, 1, 1)
        self.act = nn.GELU()
        self.conv_2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        self.p_dim = p_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        if self.training:
            x = self.act(self.conv_0(x))
            x1, x2 = torch.split(x, [self.p_dim, self.hidden_dim - self.p_dim], dim=1)  # 通道分割，减少参数量
            x1 = self.act(self.conv_1(x1))
            x = self.conv_2(torch.cat([x1, x2], dim=1))
        return x


class FIB(nn.Module):
    def __init__(self, dim=36, nlfm_model=None):
        super(FIB, self).__init__()
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.alfe = AdaptiveLFE(dim, 2)  # 使用 ALFE 捕捉局部信息
        self.nlfm = nlfm_model  # 使用 BiMamba2_2D 模型增强非局部特征的映射
        self.gelu = nn.GELU()
        self.down_scale = 8
        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))

    def forward(self, f):
        _, _, h, w = f.shape
        y, x = self.linear_0(f).chunk(2, dim=1)
        if self.nlfm is not None:
            x = self.nlfm(x)  # 使用 BiMamba2_2D 提取全局特征
        else:
            x_s = F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale))  # 如果未提供 BiMamba2_2D 模型则使用原始卷积路径
            x_v = torch.var(x, dim=(-2, -1), keepdim=True)
            x = x * F.interpolate(self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt)), size=(h, w),
                                  mode='nearest')
        y_d = self.alfe(y)  # ALFE 进行局部信息提取
        # 融合局部和全局的特征
        return self.linear_2(x + y_d)


class MCFIG(nn.Module):
    def __init__(self, dim, ffn_scale=2.0, nlfm_model=None):
        super().__init__()
        self.fib = FIB(dim, nlfm_model=nlfm_model)
        self.acfb = ACFB(dim, ffn_scale)

    def forward(self, x):
        x = self.fib(F.normalize(x)) + x
        x = self.acfb(F.normalize(x)) + x
        return x

# @ARCH_REGISTRY.register()
@ARCH_REGISTRY.register()
class MCFINet(nn.Module):
    def __init__(self, dim=32, n_blocks=8, ffn_scale=2, upscaling_factor=4):
        super().__init__()
        self.scale = upscaling_factor
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)

        # 定义 BiMamba2_2D 参数
        mamba2_args = {
            'n_layer': 12,  # Mamba-2 层数
            'd_state': 64,  # 状态维度
            'd_conv': 4,  # 卷积核大小
            'expand': 2,  # 扩展因子
            'headdim': 32,  # 头维度
            'chunk_size': 32,  # 分块大小
        }
        nlfm_model = BiMamba2_2D(cin=dim, cout=dim, d_model=dim, **mamba2_args)  # 实例化 BiMamba2_2D 模型

        self.feats = nn.Sequential(*[MCFIG(dim, ffn_scale, nlfm_model=nlfm_model) for _ in range(n_blocks)])
        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor ** 2, 3, 1, 1),
            nn.PixelShuffle(upscaling_factor)
        )

    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
        x = self.to_img(x)
        return x


# # 使用示例
# model = MCFINet(dim=64, n_blocks=8, ffn_scale=2, upscaling_factor=4)  # 构建超分模型
#
# # 输入示例
# input_tensor = torch.randn(1, 3, 64, 64)  # 假设输入为 64x64 的图像
# output = model(input_tensor)
# print("Output shape:", output.shape)  # 输出的形状