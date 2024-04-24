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

batch_size = 32
num_epochs = 20
learning_rate = 0.001

def getMinibatch(file_names):
    file_num = len(file_names)
    jmatrix_label = torch.zeros(file_num, 4, 17, 17)
    csi_data = []  
    for i, file_name in enumerate(file_names):
        img = Image.open(file_name) 
        img = img.resize((3, 30*5))  # 假设图片尺寸需要缩放到 (3, 30*5)
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float()  # 转换为张量并调整通道顺序
        csi_data.append(img_tensor)

        data = hdf5storage.loadmat(file_name, variable_names={'jointsMatrix'})
        jmatrix_label[i] = torch.from_numpy(data['jointsMatrix']).type(torch.FloatTensor)
    # 将列表转换为张量
    csi_data = torch.stack(csi_data, dim=0)
    return csi_data, jmatrix_label

mats = glob.glob('E:/Mycode/WPCP/01.mat')
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

        print(loss.item())
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    endl = time.time()
    print('Costing time:', (endl-start)/60)

torch.save(unet_model.state_dict(), 'weights/unet_model.pkl')  # 保存 U-Net 模型的参数
