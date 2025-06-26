import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# 文件夹路径
input_folder = r'G:\python_new\ggb_new\top_mu\yu_zhi\img'  # 替换为你的输入文件夹路径
output_folder = r'G:\python_new\ggb_new\top_mu\yu_zhi\label_yuzhi'  # 替换为你的输出文件夹路径

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历文件夹中的所有图像文件
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # 读取图像
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取图像

        # 二值化处理
        threshold_value = 62  # 设置阈值
        _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

        # 创建结构元素
        kernel = np.ones((3, 3), np.uint8)  # 5x5 的正方形结构元素

        # 开运算：先腐蚀后膨胀
        opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

        # 闭运算：先膨胀后腐蚀
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        # 保存处理后的图像
        output_path_closing = os.path.join(output_folder, f"{filename}")
        cv2.imwrite(output_path_closing, closing)

print("处理完成！")