import os
import cv2
import numpy as np
from PIL import Image


def add_salt_and_pepper_noise(image, noise_level=0.05):

    row, col, ch = image.shape
    num_salt = np.ceil(noise_level * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    image[coords[0], coords[1], :] = 1

    num_pepper = np.ceil(noise_level * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    image[coords[0], coords[1], :] = 0

    return image


def process_images_in_folder(folder_path):
    """
    处理文件夹中的所有图像
    :param folder_path: 图像文件夹路径
    """
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # 构造完整的文件路径
            file_path = os.path.join(folder_path, filename)

            # 打开图像
            image = Image.open(file_path)

            # 将图像转换为numpy数组
            image_np = np.array(image)

            # 翻转图像180度
            flipped_image = cv2.flip(image_np, -1)

            # 添加椒盐噪声
            noisy_image = add_salt_and_pepper_noise(flipped_image)

            # 将numpy数组转换回Pillow图像
            noisy_image_pil = Image.fromarray(noisy_image.astype('uint8'))

            # 保存处理后的图像
            output_path = os.path.join(folder_path, f"processed_{filename}")
            noisy_image_pil.save(output_path)
            print(f"Processed and saved: {output_path}")


# 替换为你的文件夹路径
folder_path = "G:\python_new\ggb_new\top_mu\m1"
process_images_in_folder(folder_path)