import scipy.io as sio
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import os
import cv2
from PIL import Image
import hdf5storage
from models.wisppn_unet import UNet
import torchvision.transforms as transforms

limb = np.array([[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12],
            [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
            [1, 3], [2, 4], [3, 5], [4, 6]])

model_path = '/home/featurize/lsy/WPCP/weights/7-26-png.pkl'
unet_model = UNet(in_channels=3, out_channels=2)
unet_model.load_state_dict(torch.load(model_path))
unet_model = unet_model.cuda().eval()

# 图片预处理
img_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图片大小
    transforms.ToTensor(),  # 将图片转换为张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])

# 加载测试数据
xy = torch.zeros(1 ,4, 17, 17)
test_mat_file = '/home/featurize/lsy/dataset/test-keypoints/000000001002.mat'
data = hdf5storage.loadmat(test_mat_file, variable_names={'jointsMatrix'})
joints_matrix = data['jointsMatrix'].transpose() # 17*17
xy[0, :, :, :] = torch.from_numpy(joints_matrix).type(torch.FloatTensor)

# 加载并预处理图片
img_file_path = "/home/featurize/lsy/dataset/test-png/000000001002.png"
# os.path.join("/raw", os.path.basename(test_mat_file).replace('.mat', '.png'))
if not os.path.exists(img_file_path):
    raise FileNotFoundError("Image file not found:", img_file_path)

img = Image.open(img_file_path).convert("RGB")
img = img_transform(img)
img = img.unsqueeze(0)  # 添加批次维度

csi_data = Variable(img.cuda())

# 推理
with torch.no_grad():
    pred_xy = unet_model(csi_data)

pred_xy = pred_xy.cpu().numpy()
# pred_xy = xy

image_center_x = 320  # 假设图像宽度为256
image_center_y = 240  # 假设图像高度为256

poseVector_x = np.zeros((1,17))
poseVector_y = np.zeros((1,17))
for index in range(17):
    poseVector_x[0,index] = pred_xy[0, 0, index, index] 
    poseVector_y[0,index] = pred_xy[0, 1, index, index] 

# 加载背景图像
jpg_file_path = "/home/featurize/lsy/dataset/test-images/000000001002.jpg"
if not os.path.exists(jpg_file_path):
    raise FileNotFoundError("JPG file not found:", jpg_file_path)

frame = Image.open(jpg_file_path)
frame = frame.resize((640, 480))
frame = np.array(frame)

# plot
plt.imshow(frame)
for i in range(len(limb)):
    plt.plot(poseVector_x[0,[limb[i, 0], limb[i, 1]]], poseVector_y[0,[limb[i, 0], limb[i, 1]]])
plt.savefig("/home/featurize/lsy/WPCP/img/7-26-png-1002.png")  # 保存图像到文件
plt.close()
