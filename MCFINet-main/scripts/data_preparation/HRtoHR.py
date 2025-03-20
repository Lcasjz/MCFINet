from PIL import Image
import os


def crop_and_save_batch(input_dir, output_dir, crop_size=(480, 480)):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录中的所有图像文件
    for filename in os.listdir(input_dir):
        # 判断文件是否为图片，且文件名符合0001到0100的格式
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and filename[:4].isdigit():
            input_image_path = os.path.join(input_dir, filename)

            # 打开原始图像
            img = Image.open(input_image_path)

            # 获取图像的宽度和高度
            img_width, img_height = img.size
            crop_width, crop_height = crop_size

            # 计算裁剪后的子图数量
            crop_counter = 1  # 裁剪图像的编号，从 001 开始
            for i in range(0, img_width - crop_width + 1, crop_width):
                for j in range(0, img_height - crop_height + 1, crop_height):
                    # 计算裁剪框的右下角位置
                    right = i + crop_width
                    bottom = j + crop_height

                    # 裁剪图像
                    cropped_img = img.crop((i, j, right, bottom))

                    # 生成裁剪后的文件名
                    base_name = os.path.splitext(filename)[0]  # 获取原图的文件名（去掉扩展名）
                    cropped_img_name = f"{base_name}_s{crop_counter:03d}.png"  # 命名格式为：0001_s001
                    cropped_img_path = os.path.join(output_dir, cropped_img_name)

                    # 保存裁剪后的图像
                    cropped_img.save(cropped_img_path)
                    print(f"Saved cropped image: {cropped_img_path}")

                    crop_counter += 1  # 递增裁剪图像的编号


# 示例：批量裁剪图像
input_dir = 'G:\data\ship2\img_HR'  # 输入图像文件夹路径
output_dir = 'G:\data\ship2\HR'  # 输出裁剪图像文件夹路径

crop_and_save_batch(input_dir, output_dir)
