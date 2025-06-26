import os
import cv2
import numpy as np
import hashlib

def image_hash(image):
    """计算图像像素级哈希值"""
    return hashlib.md5(image.tobytes()).hexdigest()

def delete_pixelwise_duplicate_labels_and_images(img_folder, label_folder):
    hash_map = {}
    deleted_count = 0

    label_files = sorted(os.listdir(label_folder))
    label_files = [f for f in label_files if f.lower().endswith(('.png', '.jpg', '.bmp'))]

    for label_file in label_files:
        label_path = os.path.join(label_folder, label_file)
        img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        h = image_hash(img)

        if h not in hash_map:
            hash_map[h] = (label_file, img)
        else:
            # 可能哈希碰撞，逐像素确认
            _, ref_img = hash_map[h]
            if img.shape == ref_img.shape and np.array_equal(img, ref_img):
                os.remove(label_path)

                # 删除对应原图
                name_no_ext = os.path.splitext(label_file)[0]
                for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
                    img_path = os.path.join(img_folder, name_no_ext + ext)
                    if os.path.exists(img_path):
                        os.remove(img_path)
                        print(f"删除重复 label：{label_file} 和原图：{name_no_ext + ext}")
                        break

                deleted_count += 1

    print(f"\n共删除 {deleted_count} 对重复标签图和原图。")

# 用法示例
if __name__ == "__main__":
    img_folder = r"D:\nx_image\test\P1256\image"
    label_folder = r"D:\nx_image\test\P1256\label"
    delete_pixelwise_duplicate_labels_and_images(img_folder, label_folder)
