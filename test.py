import os
import time
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image
import segmentation_models_pytorch as smp


# =========================== 数据集类 ===========================
class MyData(Dataset):
    def __init__(self, root_dir, sub_dir):
        self.img_dir = os.path.join(root_dir, sub_dir)
        self.img_names = os.listdir(self.img_dir)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path)
        return img, img_name

    def __len__(self):
        return len(self.img_names)


# =========================== 主逻辑 ===========================
def main():
    # 参数设置
    root_dir = r"D:\nx_image\test\test"
    input_subdir = "image"
    save_path = r"D:\nx_image\test\test\pre\3"
    model_path = r"C:\Users\admin\Downloads\UNET——2024-5-7_16-point_best_model (1).pth"
    os.makedirs(save_path, exist_ok=True)

    # 数据加载
    dataset = MyData(root_dir, input_subdir)

    # 模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1
    ).to(device)
    model.eval()

    # 图像预测
    transform = transforms.ToTensor()

    for idx, (img_pil, name) in enumerate(dataset, start=1):
        start_time = time.time()

        img_tensor = transform(img_pil)[0].unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            model.load_state_dict(
                torch.load(model_path))
            output = model(img_tensor)[0]  # shape: [1, 1024, 1024]

        save_image(output, os.path.join(save_path, name))

        elapsed = time.time() - start_time
        print(f"[{idx:03d}] Processed '{name}' in {elapsed:.2f} s")

if __name__ == '__main__':
    main()
