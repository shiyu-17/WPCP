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

model_path = 'weights/unet_model.pkl'
unet_model = UNet(in_channels=3, out_channels=2)
unet_model.load_state_dict(torch.load(model_path))
unet_model = unet_model.cuda().eval()

# 图片预处理
img_transform = transforms.Compose([
    # transforms.Resize((256, 256)),  # 调整图片大小
    transforms.ToTensor(),  # 将图片转换为张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])

# 加载测试数据
test_mat_file = 'raw/hbp01.mat'
data = hdf5storage.loadmat(test_mat_file, variable_names={'frame'})
frame = data['frame']

# 加载并预处理图片
img_file_path = "raw/hbp01.png"
# os.path.join("/raw", os.path.basename(test_mat_file).replace('.mat', '.png'))
if not os.path.exists(img_file_path):
    raise FileNotFoundError("Image file not found:", img_file_path)

img = Image.open(img_file_path).convert("RGB")
img = img_transform(img)
img = img.unsqueeze(0)  # 添加批次维度

# 转换为Variable并移到GPU
csi_data = Variable(img.cuda())

# 推理
with torch.no_grad():
    pred_xy = unet_model(csi_data)

# 将预测结果转换为numpy数组
pred_xy = pred_xy.cpu().numpy()

# 提取关节点坐标
poseVector_x = np.zeros((17,))
poseVector_y = np.zeros((17,))
for index in range(17):
    poseVector_x[index] = pred_xy[0, 0, index, index]
    poseVector_y[index] = pred_xy[0, 1, index, index]

# 显示图像并绘制关节点和肢体连接
plt.imshow(cv2.resize(frame, (1280, 720)))
for i in range(len(limb)):
    plt.plot(poseVector_x[[limb[i, 0], limb[i, 1]]]* frame.shape[1] / 256, poseVector_y[[limb[i, 0], limb[i, 1]]]* frame.shape[0] / 256, marker='o')
plt.show()

# 确保cv2窗口关闭
cv2.destroyAllWindows()
