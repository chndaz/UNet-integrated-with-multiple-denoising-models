import os

import cv2
import numpy
from PIL import Image
from matplotlib import pyplot as plt
for i in range(136):
    i=i+1
    dir1=fr'D:\python_new\ggb\best\test_image\333 ({i}).png'
    img = cv2.imread(dir1, 0)
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th2=numpy.array(th2)
    image = Image.fromarray(th2)  # 输出的是4通道图
    image = image.convert('L')
    resultPath = r'D:\python_new\ggb\best_new\大津法'
    image.save(os.path.join(resultPath, f'333 ({i})') + '.png')  # 前面列表遍历还有ecoph+=1每轮次数

