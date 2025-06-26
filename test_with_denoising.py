
import numpy

from torchvision.utils import save_image

import torch.nn.functional as F
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Flatten
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from PIL import Image
import segmentation_models_pytorch as smp


model = smp.Unet(
    encoder_name="mobilenet_v2",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1,
)



def contrast_and_brightness(img, alpha=1.2, beta=1 - 1.2, gamma=50):
    blank = np.zeros_like(img)
    return cv2.addWeighted(img, alpha, blank, beta, gamma)


def hough_circle_detection(image, circles):
    refined = []
    radius_threshold = 45 if circles[0][2] >= 9 else 30

    for c in circles:
        x1, y1 = int(c[0]) - radius_threshold, int(c[1]) - radius_threshold
        x2, y2 = int(c[0]) + radius_threshold, int(c[1]) + radius_threshold
        x1, y1 = max(0, x1), max(0, y1)

        nearby = [
            int(j[0]) for j in circles
            if x1 < j[0] < x2 and y1 < j[1] < y2
        ]
        if len(nearby) >= 9:
            refined.append([int(c[0]), int(c[1]), int(c[2])])

    # 去重
    unique = []
    for r in refined:
        if r not in unique:
            unique.append(r)

    return np.array(unique)


def apply_hough_transform(image):
    enhanced = contrast_and_brightness(image)
    circles = cv2.HoughCircles(enhanced, cv2.HOUGH_GRADIENT, 1, 15,
                                param1=200, param2=10, minRadius=5, maxRadius=10)

    if circles is not None:
        circles = hough_circle_detection(enhanced, circles[0])
        for c in circles:
            cv2.circle(image, (c[0], c[1]), c[2], (255, 255, 255), 40)

    img = Image.fromarray(image).convert('L')
    return img


def flatten_binary(img_tensor):
    return torch.where(img_tensor[0] < 0.4, 0.0, 1.0).reshape(1024, 1024)


def coordinate_mask(mask):
    coords = torch.stack(torch.meshgrid(torch.arange(1024), torch.arange(1024), indexing='ij'), dim=-1)
    coords = torch.cat([mask.unsqueeze(-1), coords], dim=-1)  # shape: [1024, 1024, 3]
    return coords


def restore_internal_area(coord_mask, original_img):
    result = []
    original_img = original_img.reshape(1024, 1024)
    original_img=numpy.array(original_img)
    coord_mask = numpy.array(coord_mask)

    for coord, pixel in zip(coord_mask.reshape(-1, 3), original_img.flatten()):
        result.append(pixel if coord[0] == 0 else coord[0])

    return torch.tensor(result, dtype=torch.float32).reshape(1024, 1024)


def prod_to_tensor(img_tensor):
    binary = (img_tensor[0] >= 0.4).float()
    return binary.reshape(1024, 1024)


def edge_check(img_patch):
    edges = []
    h, w = img_patch.shape

    for i in range(h):
        if i == 0 or i == h - 1:
            edges.extend(img_patch[i])
        else:
            edges.append(img_patch[i][0])
            edges.append(img_patch[i][w - 1])
    return edges


def patch_fill(img, patch_h=16, patch_w=16):
    h, w = img.shape
    img_np = img.numpy()
    filled = np.zeros((h, w))

    for i in range(0, h, 2):
        for j in range(0, w, 2):
            if i + patch_h < h and j + patch_w < w:
                patch = img_np[i:i + patch_h, j:j + patch_w]
                if 0 not in edge_check(patch):
                    filled[i:i + patch_h, j:j + patch_w] = 1

    filled = torch.tensor(filled)
    return restore_internal_area(coordinate_mask(filled), img)


# =========================== 数据集类 ===========================
class MyData(Dataset):
    def __init__(self, img_dir ):
        self.img_dir = img_dir
        self.img_names = os.listdir(self.img_dir)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path)
        return img, img_name

    def __len__(self):
        return len(self.img_names)


def median_filter_3x3(input_tensor):
    # 使用 F.pad 对输入张量进行填充，以保持边界
    padded = F.pad(input_tensor, (1, 1, 1, 1), mode='replicate')

    # 使用 F.unfold 将 3x3 的邻域展开为一个矩阵
    unfolded = F.unfold(padded, kernel_size=3)

    # 重塑展开后的张量，以便进行中值计算
    unfolded = unfolded.reshape(input_tensor.shape[0], 9, -1)

    # 计算中值
    median = torch.median(unfolded, dim=1).values

    # 重塑结果为原始输入张量的形状
    median = median.reshape(input_tensor.shape)

    return median

# =========================== 主逻辑 ===========================
def main(save_path ):
    # 参数设置
    root_dir = r"D:\liangX_python_new_code\python_new_code\ggb_new\segmentation\m3"
    # input_subdir = "image"
    model_path = r"D:\liangX_python_new_code\python_new_code\ggb_new\segmentation\mobiv2unet_25_06_01_21.pth"
    os.makedirs(save_path, exist_ok=True)

    # 数据加载
    dataset = MyData(root_dir)

    # 模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1
    ).to(device)

    # 图像预测
    transform = transforms.ToTensor()

    for idx, (img_pil, name) in enumerate(dataset, start=1):
        start_time = time.time()

        img_tensor = transform(img_pil)[0].unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            model.load_state_dict(
                torch.load(model_path))
            output = model(img_tensor)[0][0]  # shape: [1, 1024, 1024]
        output=output.detach().cpu().numpy()

        image_np = np.array(img_pil)
        processed = apply_hough_transform(image_np)
        processed_tensor = transforms.ToTensor()(processed)
        binary_mask = prod_to_tensor(processed_tensor)
        coord_mask = coordinate_mask(binary_mask)
        restored = restore_internal_area(coord_mask, output)
        restored = restored.unsqueeze(0).unsqueeze(0)  # 添加批量维度和通道维度
        filtered_a = median_filter_3x3(restored)
        filtered_a = filtered_a.clone().detach().reshape(1024, 1024)


        save_image(filtered_a, os.path.join(save_path, name))
if __name__ == '__main__':
    save_path= r"D:\liangX_python_new_code\python_new_code\ggb_new\segmentation\output1"
    main(save_path)
