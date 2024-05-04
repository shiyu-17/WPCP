import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
# import matplotlib.pyplot as plt
import math
import time
import sys
import glob
import hdf5storage
from random import shuffle
import time
import os
import torch.nn as nn
from torchsummary import summary
from PIL import Image

from models.wisppn_unet import UNet 
import torchvision.transforms as transforms 

batch_size = 32
num_epochs = 20
learning_rate = 0.001

height = 256
width = 256

# 图片预处理 120*120
img_transform = transforms.Compose([
    # transforms.Resize((height, width)),  # 调整图片大小
    transforms.ToTensor(),  # 将图片转换为张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])

def getMinibatch(file_names):
    file_num = len(file_names)
    jmatrix_label = torch.zeros(file_num, 4, 17, 17)
    csi_data = []  
    for i, file_name in enumerate(file_names):
        print(file_name)
        # 构建图片文件路径
        img_file_path = os.path.join("/user90/djy/lsy/wpcp_data/train/csi_pic", os.path.basename(file_name).replace('.mat', '.png'))
        if not os.path.exists(img_file_path):
            raise FileNotFoundError("Image file not found:", img_file_path)
        img = Image.open(img_file_path).convert("RGB")
        
        # 图片预处理
        img = img_transform(img)
        csi_data.append(img)

        data = hdf5storage.loadmat(file_name, variable_names={'jointsMatrix'})
        joints_matrix = data['jointsMatrix'].transpose()
        jmatrix_label[i] = torch.from_numpy(joints_matrix).type(torch.FloatTensor)
    # 将列表转换为张量
    csi_data = torch.stack(csi_data, dim=0)
    return csi_data, jmatrix_label

mats = glob.glob('/user90/djy/lsy/wpcp_data/train/keypoints/*.mat')
print("Total number of samples:", len(mats))
mats_num = len(mats)
batch_num = int(np.floor(mats_num/batch_size))

unet_model = UNet(in_channels=3, out_channels=2)  
unet_model = unet_model.cuda()

criterion_L2 = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(unet_model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 20, 25, 30], gamma=0.5)

unet_model.train()

for epoch_index in range(num_epochs):
    scheduler.step()
    print("Epoch:", epoch_index + 1)
    print("Current Learning Rate:", scheduler.get_lr())
    start = time.time()
    # shuffling dataset
    shuffle(mats)
    loss_x = 0
    # in each minibatch
    for batch_index in range(batch_num):
        if batch_index < batch_num:
            file_names = mats[batch_index*batch_size:(batch_index+1)*batch_size]
        else:
            file_names = mats[batch_num*batch_size:]

        csi_data, jmatrix_label = getMinibatch(file_names)

        csi_data = Variable(csi_data.cuda())
        xy = Variable(jmatrix_label[:, 0:2, :, :].cuda())
        confidence = Variable(jmatrix_label[:, 2:4, :, :].cuda())

        pred_xy = unet_model(csi_data)
        loss = criterion_L2(torch.mul(confidence, pred_xy), torch.mul(confidence, xy))

        print("Batch [{} / {}], Loss: {:.6f}".format(batch_index + 1, batch_num, loss.item()))
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    endl = time.time()
    print('Epoch Time:', (endl - start) / 60, 'minutes')

torch.save(unet_model.state_dict(), '/user90/djy/lsy/WPCP/weights/unet_model.pkl')  # 保存 U-Net 模型的参数
