import datetime
import logging
import math
import time
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options




def init_tb_loggers(opt):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters


def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt, resume_state['iter'])
    return resume_state


def train_pipeline(root_path):
    # parse options, set distributed setting, set random seed
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    # create model
    model = build_model(opt)
    if resume_state:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer.record()

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_timer.record()
            if current_iter == 1:
                # reset start time in msg_logger for more accurate eta_time
                # not work in resume mode
                msg_logger.reset_start_time()
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # validation
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                if len(val_loaders) > 1:
                    logger.warning('Multiple validation datasets are *only* supported by SRModel.')
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()
        # end of iter

    # end of epoch

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()


# import datetime
# import logging
# import math
# import time
# import torch
# from os import path as osp
#
# from basicsr.data import build_dataloader, build_dataset
# from basicsr.data.data_sampler import EnlargedSampler
# from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
# from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
#                            init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
# from basicsr.utils.options import copy_opt_file, dict2str, parse_options
#
# def train_pipeline(root_path):
#     # 解析选项，设置分布式配置并设置随机种子
#     opt, args = parse_options(root_path, is_train=True)
#     opt['root_path'] = root_path
#
#     torch.backends.cudnn.benchmark = True
#
#     # 加载恢复状态（如果需要）
#     resume_state = load_resume_state(opt)
#     if resume_state is None:
#         make_exp_dirs(opt)
#         if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
#             mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))
#
#     copy_opt_file(args.opt, opt['path']['experiments_root'])
#
#     # 初始化日志
#     log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
#     logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
#     logger.info(get_env_info())
#     logger.info(dict2str(opt))
#     tb_logger = init_tb_loggers(opt)
#
#     # 创建训练和验证数据加载器
#     result = create_train_val_dataloader(opt, logger)
#     train_loader, train_sampler, val_loaders, total_epochs, total_iters = result
#
#     # 使用build_model(opt)动态构建模型
#     model = build_model(opt)  # 这里根据yml文件中的配置构建模型
#
#     # 确认net_g存在
#     if hasattr(model, 'net_g'):  # 这里使用net_g，而不是network_g
#         optimizer = torch.optim.Adam(model.net_g.parameters(), lr=opt['train']['optim_g']['lr'])
#     else:
#         raise AttributeError('The model does not have net_g attribute.')
#
#     if resume_state:  # 恢复训练
#         model.resume_training(resume_state)  # 处理优化器和调度器
#         logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
#         start_epoch = resume_state['epoch']
#         current_iter = resume_state['iter']
#     else:
#         start_epoch = 0
#         current_iter = 0
#
#     msg_logger = MessageLogger(opt, current_iter, tb_logger)
#
#     prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
#     if prefetch_mode is None or prefetch_mode == 'cpu':
#         prefetcher = CPUPrefetcher(train_loader)
#     elif prefetch_mode == 'cuda':
#         prefetcher = CUDAPrefetcher(train_loader, opt)
#         logger.info(f'Use {prefetch_mode} prefetch dataloader')
#         if opt['datasets']['train'].get('pin_memory') is not True:
#             raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
#     else:
#         raise ValueError(f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'.")
#
#     # 训练
#     logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
#     data_timer, iter_timer = AvgTimer(), AvgTimer()
#     start_time = time.time()
#
#     for epoch in range(start_epoch, total_epochs + 1):
#         train_sampler.set_epoch(epoch)
#         prefetcher.reset()
#         train_data = prefetcher.next()
#
#         while train_data is not None:
#             data_timer.record()
#
#             current_iter += 1
#             if current_iter > total_iters:
#                 break
#
#             model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
#             model.feed_data(train_data)
#             model.optimize_parameters(current_iter)
#
#             # 获取输入和目标数据
#             input_img = train_data['lq']  # 低质量图像作为输入
#             target_img = train_data['gt']  # 高质量图像作为目标
#
#             # 使用模型的 test 方法进行前向传播
#             model.feed_data(train_data)  # 传入数据
#             model.test()  # 执行前向传播
#             output_img = model.get_current_visuals()['result']  # 获取输出图像
#
#             # 动态稀疏率调整（如果存在LW_SMFA模块）
#             if hasattr(model.net_g, 'lw_smfa'):  # 假设net_g包含LW_SMFA
#                 model.net_g.lw_smfa.adjust_sparsity_ratio(output_img, target_img)  # 动态调整稀疏率
#
#             iter_timer.record()
#             if current_iter == 1:
#                 msg_logger.reset_start_time()
#
#             # 打印日志
#             if current_iter % opt['logger']['print_freq'] == 0:
#                 log_vars = {'epoch': epoch, 'iter': current_iter}
#                 log_vars.update({'lrs': model.get_current_learning_rate()})
#                 log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
#                 log_vars.update(model.get_current_log())
#                 msg_logger(log_vars)
#
#             # 保存模型和训练状态
#             if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
#                 logger.info('Saving models and training states.')
#                 model.save(epoch, current_iter)
#
#             # 验证
#             if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
#                 for val_loader in val_loaders:
#                     model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
#
#             data_timer.start()
#             iter_timer.start()
#             train_data = prefetcher.next()
#
#     # 训练结束，保存最新模型
#     consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
#     logger.info(f'End of training. Time consumed: {consumed_time}')
#     logger.info('Save the latest model.')
#     model.save(epoch=-1, current_iter=-1)
#
#     if opt.get('val') is not None:
#         for val_loader in val_loaders:
#             model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
#
#     if tb_logger:
#         tb_logger.close()








from basicsr.utils.registry import MODEL_REGISTRY, ARCH_REGISTRY



if __name__ == '__main__':

    # 打印 MODEL_REGISTRY 中所有注册的对象
    print("Registered objects in MODEL_REGISTRY:")
    for name, obj in MODEL_REGISTRY:
        print(f"Name: {name}, Object: {obj}")

    # 打印 ARCH_REGISTRY 中所有注册的对象
    print("\nRegistered objects in ARCH_REGISTRY:")
    for name, obj in ARCH_REGISTRY:
        print(f"Name: {name}, Object: {obj}")

    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))

    train_pipeline(root_path)
