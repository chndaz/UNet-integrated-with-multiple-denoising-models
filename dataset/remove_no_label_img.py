#
#
#
#
# import os
#
# def find_and_delete_unlabeled_images(img_folder, label_folder):
#     img_files = os.listdir(img_folder)
#     label_files = set(os.listdir(label_folder))  # 提高查找效率
#
#     deleted_count = 0
#     for img_file in img_files:
#         # 获取图像文件名（无扩展名）并构造对应的 label 文件名
#         name_without_ext = os.path.splitext(img_file)[0]
#         label_file = name_without_ext + '.txt'
#
#         if label_file not in label_files:
#             # 找不到对应标签，删除图像
#             img_path = os.path.join(img_folder, img_file)
#             try:
#                 # os.remove(img_path)
#                 print(f"已删除未标注图像：{img_file}")
#                 deleted_count += 1
#             except Exception as e:
#                 print(f"删除失败：{img_file}，错误信息：{e}")
#
#     if deleted_count == 0:
#         print("所有图像都有对应标签文件，无需删除。")
#     else:
#         print(f"\n共删除 {deleted_count} 张没有标签的图像。")
#
# def main():
#     img_folder = r"G:\yolov5\data\icimg_data\train_nx\train\image"  # 替换为你的 img 文件夹路径
#     label_folder = r"G:\yolov5\data\icimg_data\train_nx\train\label"  # 替换为你的 label 文件夹路径
#
#     find_and_delete_unlabeled_images(img_folder, label_folder)
#
# if __name__ == "__main__":
#     main()


import os
import shutil


def delete_images_without_labels(img_folder, label_folder):
    """
    删除 img 文件夹中没有对应标签文件的图像。

    参数:
        img_folder (str): 图像文件夹路径
        label_folder (str): 标签文件夹路径
    """
    # 获取 img 文件夹中的所有文件名
    img_files = os.listdir(img_folder)

    # 获取 label 文件夹中的所有文件名
    label_files = os.listdir(label_folder)

    # 创建一个集合，存储所有标签文件的名称（去掉扩展名）
    label_names = {os.path.splitext(label)[0] for label in label_files}

    # 遍历图像文件夹
    for img_file in img_files:
        # 获取图像文件的名称（去掉扩展名）
        img_name = os.path.splitext(img_file)[0]

        # 检查是否存在对应的标签文件
        if img_name not in label_names:
            # 如果没有对应的标签文件，删除该图像文件
            img_path = os.path.join(img_folder, img_file)
            print(f"Deleting image without label: {img_path}")
            os.remove(img_path)


img_folder = r"D:\nx_image\test\P1256\image"  # 替换为你的 img 文件夹路径
label_folder = r"D:\nx_image\test\P1256\label" # 替换为你的 label 文件夹路径

# 调用函数
delete_images_without_labels(img_folder, label_folder)