# UNetplusplus-integrated-with-multiple-denoising-models
This study proposes a combined approach of segmentation
 and post- processing. During the segmentation stage, we use UNet++ as the base architecture, with EfficientNet- B7 
as the encoder, resulting in an E- UNet++ model. This model effectively combines the efficiency and pre- training capabilities of 
EfficientNet with the ability of UNet++ to capture both global structural information and fine- grained boundary details in IC 
images, enabling it to effectively handle challenges such as high resolution and limited training samples. In the post- processing 
stage, to address potential noise caused by the insufficient utilization of spatial location information in network- based methods, 
we propose the use of Hough circle detection and median filtering to eliminate noise from vias and non- via regions.
## Work Published in following article
Cheng H, Yu C, Zhang C. Segmentation of IC Images in Integrated Circuit Reverse Engineering Using EfficientNet Encoder Based on U‐Net++ Architecture[J]. International Journal of Circuit Theory and Applications, 2025.https://doi.org/10.1002/cta.4485
## Folder
Folder consists of 

1.Dataset folder contains some examples of IC images.

2.Model folder contains the existing model.

3.Results folder contains the visualized results of the proposed model.
## Details   
The file Clip_IC_image_256.py in the dataset directory is primarily responsible for cropping 1024×1024 IC images into 256×256 patches. This preprocessing step facilitates subsequent data augmentation operations. The script dataset_split.py is used to divide the images and their corresponding labels in the dataset into training, testing, and validation subsets. The purpose of remove_black.py is to eliminate IC images that share identical label masks, thereby removing redundant samples from the dataset. Lastly, remove_no_label_img.py is designed to discard IC images that do not have corresponding annotation labels, ensuring consistency between image and label pairs in the dataset.
## Preparing Data  
Download dataset, i.e.https://doi.org/10.17617/3.HY5SYN
