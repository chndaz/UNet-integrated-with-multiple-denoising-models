import os


def add_prefix_to_filenames(folder_path, prefix="P1256_"):
    """
    给指定文件夹中的所有文件名前面加上指定的前缀。

    参数:
        folder_path (str): 文件夹路径
        prefix (str): 要添加的前缀，默认为 "m2_"
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"文件夹 {folder_path} 不存在！")
        return

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 构造原始文件的完整路径
        old_file_path = os.path.join(folder_path, filename)

        # 检查是否是文件（而不是文件夹）
        if os.path.isfile(old_file_path):
            # 构造新的文件名（加上前缀）
            new_filename = f"{prefix}{filename}"
            new_file_path = os.path.join(folder_path, new_filename)

            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f"文件 {filename} 已重命名为 {new_filename}")

x_m="4_8623"
# 设置文件夹路径
img_folder = fr"D:\nx_image\test\{x_m}\image"  # 替换为你的 img 文件夹路径
label_folder = fr"D:\nx_image\test\{x_m}\label" # 替换为你的 label 文件夹路径

# 调用函数
add_prefix_to_filenames(img_folder,prefix=f'{x_m}_')
add_prefix_to_filenames(label_folder,prefix=f'{x_m}_')