import os
import shutil
import random
from tqdm import tqdm


def split_dataset(base_dir, image_dir='image', label_dir='label', ratios=(0.8, 0.1, 0.1)):
    """
    分割数据集到train/val/test文件夹

    参数:
        base_dir: 包含image和label文件夹的根目录
        image_dir: 图像文件夹名 (默认'image')
        label_dir: 标签文件夹名 (默认'label')
        ratios: 分割比例 (train, val, test) 默认(0.8, 0.1, 0.1)
    """
    # 检查目录结构
    image_path = os.path.join(base_dir, image_dir)
    label_path = os.path.join(base_dir, label_dir)

    if not os.path.exists(image_path) or not os.path.exists(label_path):
        raise ValueError(f"必须在{base_dir}下同时存在{image_dir}和{label_dir}文件夹")

    # 获取所有图像文件(假设图像和标签文件名一一对应)
    image_files = sorted([f for f in os.listdir(image_path) if not f.startswith('.')])
    label_files = sorted([f for f in os.listdir(label_path) if not f.startswith('.')])

    # 验证文件对应关系
    assert len(image_files) == len(label_files), "图像和标签文件数量不匹配"
    for img, lbl in zip(image_files, label_files):
        assert os.path.splitext(img)[0] == os.path.splitext(lbl)[0], f"文件名不匹配: {img} vs {lbl}"

    # 打乱文件顺序(固定随机种子保证可重复性)
    random.seed(42)
    combined = list(zip(image_files, label_files))
    random.shuffle(combined)
    image_files, label_files = zip(*combined)

    # 计算分割点
    total = len(image_files)
    train_end = int(total * ratios[0])
    val_end = train_end + int(total * ratios[1])

    splits = {
        'train': (0, train_end),
        'val': (train_end, val_end),
        'test': (val_end, total)
    }

    # 创建目标文件夹
    for split in splits:
        for folder in [image_dir, label_dir]:
            os.makedirs(os.path.join(base_dir, split, folder), exist_ok=True)

    # 复制文件
    for split, (start, end) in splits.items():
        print(f"正在处理 {split} 数据集...")
        for img_file, lbl_file in tqdm(zip(image_files[start:end], label_files[start:end])):
            # 复制图像
            src_img = os.path.join(image_path, img_file)
            dst_img = os.path.join(base_dir, split, image_dir, img_file)
            shutil.copy2(src_img, dst_img)

            # 复制标签
            src_lbl = os.path.join(label_path, lbl_file)
            dst_lbl = os.path.join(base_dir, split, label_dir, lbl_file)
            shutil.copy2(src_lbl, dst_lbl)

    print("数据集分割完成！")


if __name__ == '__main__':
    # 使用示例 - 修改为你的实际路径
    dataset_path = r'E:\nx_ic_image\second_method\train'  # 替换为你的数据集路径
    split_dataset(dataset_path)