import time
import torchvision
from torch import nn
import copy
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import segmentation_models_pytorch as smp

model = smp.UnetPlusPlus(
    encoder_name="mobilenet_v2",
    encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1  # model output channels (number of classes in your dataset)
)


model_name='mobiUnetplus_0625_11_'
# #####################################
tran=transforms.ToTensor()


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience  # 容忍多少轮没有提升
        self.min_delta = min_delta  # 最小提升量
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"早停计数器：{self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True




class Dataset1(Dataset):
    # 初始化类 根据类创建实例时要运行函数，为整个class提供全局变量
    def __init__(self, root_dir,value_dir,label_dir,h_and_w):
        self.root_dir = root_dir  # 函数的变量不能传递给另外一个变量，而self能够把指定变量给别的函数使用，全局变量
        self.label_dir = label_dir
        self.value_dir=value_dir
        self.path2= os.path.join(self.root_dir, self.label_dir) # 路径的拼接
        self.path1= os.path.join(self.root_dir, self.value_dir) # 路径的拼接
        self.img_path1 = os.listdir(self.path1)  # 获得图片所有地址
        self.img_path2 = os.listdir(self.path1)  # 获得图片所有地址
        self.h_and_w=h_and_w
    ## 获取所有图片的地址列表
    def __getitem__(self, idx):
        img_name1 = self.img_path1[idx] #获取图片名称  self.全局的
        img_name2= self.img_path2[idx] #获取图片名称  self.全局的
        img_item_path = os.path.join(self.path1, img_name1) # 获取每个图片的地址(相对路径)
        smg_item_path = os.path.join(self.path2, img_name2) # 获取每个图片的地址(相对路径)
        img1= Image.open(img_item_path)
        img2=Image.open(smg_item_path)
        img1=tran(img1)
        img2=tran(img2)
        # print(img1.shape)
        # print(img2.shape)
        img1=img1[0]
        img2= img2[0]
        img1=torch.reshape(img1,(1,self.h_and_w,self.h_and_w))
        img2=torch.reshape(img2,(1,self.h_and_w,self.h_and_w))
        return img1,img2

    def __len__(self):
        return len(self.img_path1)#这里返回一个就行


data=Dataset1(r"G:\python_new\ggb_new\segmentation\cunchu\data_enhancement\data_enhancement_new\data_enhancement\image","train_image_clip_image","train_image_clip_label",h_and_w=256)
data2=Dataset1(r"G:\python_new\ggb_new\segmentation\cunchu\data_enhancement\data_enhancement_new\data_enhancement\image","val_image","val_label",h_and_w=1024)
data2_loader=DataLoader(data2,batch_size=2,shuffle=True)
data_loader=DataLoader(data,batch_size=2,shuffle=True)


class DoubleConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
    def forward(self, x: torch.Tensor):
        x = self.first(x)
        x = self.act1(x)
        x = self.second(x)
        return self.act2(x)
class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self,x:torch.Tensor):
        return self.pool(x)
class UpSample(nn.Module):
    def __init__(self,input_channals:int,output_channals:int):
        super().__init__()
        self.up = nn.ConvTranspose2d(input_channals,output_channals,kernel_size=2,stride=2)
        #看效果，不好试试UpsamplingBilinear2d(scale_factor=2)
    def forward(self,x:torch.Tensor):
        return self.up(x)
class CropAndConcat(nn.Module):

    def forward(self,x:torch.Tensor,contracting_x:torch.Tensor):
        contracting_x = torchvision.transforms.functional.center_crop(contracting_x,[x.shape[2],x.shape[3]])
        x = torch.cat([x,contracting_x],dim=1)
        return x

def train_process(model, data_train, data_test,num_epoch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = nn.BCEWithLogitsLoss()
    model = model.to(device)
    # 复制当前模型的参数
    best_model = copy.deepcopy(model.state_dict())
    # 训练集损失函数的列表
    train_loss_all = []
    # 验证集损失函数列表
    val_loss_all = []
    # 计时(当前时间)
    since = time.time()
    k=1
    best_loss = float('inf')
    #squeeze函数会把前面四则（1，*，*）变为（*，*）
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    for i in range(num_epoch):
        print(f'第{k}轮开始,总共{num_epoch}轮')
        #初始化值
        train_loss=0
        #训练集准确度
        train_correct=0
        val_loss=0
        #验证集的准确度
        val_correct=0
        train_num=0
        val_num=0
        plt.figure(figsize=(12,5))
        for batch_idx, (image,label) in enumerate(tqdm(data_train, desc="训练中")):
            image=image.to(device)
            label=label.to(device)
            #训练模式
            model.train()
            output=model(image)
            # pre_label=torch.argmax(output,dim=1)
            loss_train=loss(output,label)
            optim.zero_grad()
            #这里的loss_train为64个样本的平均值
            loss_train.backward()

            optim.step()
            train_loss+=loss_train.item()*image.size(0)#总的样本loss的累加
            train_correct+=torch.sum(output==label)
            train_num+=image.size(0)
        for jp in data_test:
            image1,label1=jp
            image1=image1.to(device)
            label1=label1.to(device)
            #评估模式
            model.eval()
            output1=model(image1)
            # pre_label_test=torch.argmax(output,dim=1)
            loss_test=loss(output1,label1)
            #对损失函数进行累加
            val_loss+=loss_test.item()*image.size(0)#这里乘以64了
            val_correct+=torch.sum(output==label)
            val_num+=image.size(0)

        #该轮次平均的loss
        train_loss_all.append(train_loss/train_num)
        val_loss_all.append(val_loss/val_num)

        if val_loss_all[-1]<best_loss:
            best_loss=val_loss_all[-1]
            print("best_loss=",best_loss)
            #保存参数
            best_acc_wts=copy.deepcopy(model.state_dict())
            torch.save(best_acc_wts, f'{model_name}_{k}best_model25_0623.pth')
        #时间
        # 添加 EarlyStopping 检查
        early_stopping(val_loss_all[-1])
        if early_stopping.early_stop:
            print("验证集损失无提升，触发早停！")
            break
        time_use=time.time()-since
        print(f'训练总耗费时间{time_use//60}m,{time_use%60}s')
        k+=1
    #选择最优参数
    #选择最高精确度的模型参数
    torch.save(best_acc_wts,fr'G:\python_new\ggb_new\cd\pth\{model_name}_MOBI_25_.pth')



if __name__ == '__main__':
    #模型实例化
    train_process=train_process(model,data_loader,data2_loader,num_epoch=10)