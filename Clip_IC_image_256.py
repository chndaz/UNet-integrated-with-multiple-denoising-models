import cv2
import os
import numpy as np


def crop_images(input_img_folder, input_label_folder, output_img_folder, output_label_folder, crop_size=256):
    # 确保输出文件夹存在
    os.makedirs(output_img_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)

    # 获取图像和标签文件列表
    img_files = sorted(os.listdir(input_img_folder))
    label_files = sorted(os.listdir(input_label_folder))

    # 确保图像和标签文件一一对应
    assert len(img_files) == len(label_files), "图像和标签数量不匹配"

    for img_file, label_file in zip(img_files, label_files):
        # 读取图像和标签
        img_path = os.path.join(input_img_folder, img_file)
        label_path = os.path.join(input_label_folder, label_file)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        # 获取文件名和扩展名
        img_name, img_ext = os.path.splitext(img_file)
        label_name, label_ext = os.path.splitext(label_file)

        # 检查图像尺寸是否为1024x1024
        assert img.shape == (1024, 1024), f"图像 {img_file} 尺寸不是1024x1024"
        assert label.shape == (1024, 1024), f"标签 {label_file} 尺寸不是1024x1024"

        # 裁剪图像和标签
        for i in range(0, 1024, crop_size):
            for j in range(0, 1024, crop_size):
                # 裁剪图像
                img_crop = img[i:i + crop_size, j:j + crop_size]
                # 裁剪标签
                label_crop = label[i:i + crop_size, j:j + crop_size]

                # 生成保存文件名
                crop_idx = (i // crop_size) * 4 + (j // crop_size)  # 0-15的索引
                img_crop_name = f"{img_name}_{crop_idx}{img_ext}"
                label_crop_name = f"{label_name}_{crop_idx}{label_ext}"

                # 保存裁剪后的图像和标签
                cv2.imwrite(os.path.join(output_img_folder, img_crop_name), img_crop)
                cv2.imwrite(os.path.join(output_label_folder, label_crop_name), label_crop)


# 使用示例
input_img_folder = r"E:\nx_ic_image\big_lib\gray_level\image"
input_label_folder = r"E:\nx_ic_image\big_lib\gray_level\label"
output_img_folder = r"E:\nx_ic_image\big_lib\gray_level\image_clip" 
output_label_folder = r"E:\nx_ic_image\big_lib\gray_level\label_clip"

crop_images(input_img_folder, input_label_folder, output_img_folder, output_label_folder)