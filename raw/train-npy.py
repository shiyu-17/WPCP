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
import os
import torch.nn as nn
from torchsummary import summary
from PIL import Image
from models.wisppn_unet import UNet 
import torchvision.transforms as transforms 
import logging

batch_size = 16
num_epochs = 20
learning_rate = 0.01

height = 256
width = 256

# 设置日志文件保存目录
log_dir = "/home/featurize/lsy/WPCP/logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 设置日志文件名，使用当前时间命名
log_filename = os.path.join(log_dir, time.strftime("%Y%m%d-%H%M%S") + ".txt")
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

# 图片预处理 120*120
img_transform = transforms.Compose([
    transforms.Resize((height, width)),  # 调整图片大小
    transforms.ToTensor(),  # 将图片转换为张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])

def getMinibatch(file_names):
    file_num = len(file_names)
    jmatrix_label = torch.zeros(file_num, 4, 17, 17)
    csi_data = []  
    for i, file_name in enumerate(file_names):
        # 构建图片文件路径
        npy_file_path = os.path.join("/home/featurize/lsy/dataset/train-npy", os.path.basename(file_name).replace('.mat', '.npy'))
        if not os.path.exists(npy_file_path):
            raise FileNotFoundError("Image file not found:", npy_file_path)
        
        # 判断文件扩展名并相应处理
        if npy_file_path.endswith('.npy'):
            img = np.load(npy_file_path)
            if img.shape != (3, 800, 800):
                raise ValueError(f"Unexpected image shape: {img.shape}")
            # 调整维度顺序
            img = img.transpose(1, 2, 0)
            img = Image.fromarray((img * 255).astype('uint8'))
            img = img_transform(img)
        else:
            img = Image.open(npy_file_path).convert("RGB")
            img = img_transform(img)

        csi_data.append(img)

        data = hdf5storage.loadmat(file_name, variable_names={'jointsMatrix'})
        # [x; y; c; c] 17*17*4
        # jmatrix_label[i, :, :, :] = torch.from_numpy(data['jointsMatrix']).type(torch.FloatTensor)
        joints_matrix = data['jointsMatrix'].transpose(2, 0, 1) # 4*17*17
        jmatrix_label[i, :, :, :] = torch.from_numpy(joints_matrix).type(torch.FloatTensor)

    for img in csi_data:
        assert img.size() == (3, height, width), f"Image size mismatch: {img.size()}"

    # 将列表转换为张量
    csi_data = torch.stack(csi_data, dim=0)
    #print (csi_data.shape) 
    return csi_data, jmatrix_label

mats = glob.glob('/home/featurize/lsy/dataset/train-keypoints/*.mat')
logging.info("Total number of samples: {}".format(len(mats)))
mats_num = len(mats)
batch_num = int(np.floor(mats_num/batch_size))

unet_model = UNet(in_channels=3, out_channels=2)  
unet_model = unet_model.cuda()

criterion_L2 = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(unet_model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 20, 25, 30], gamma=0.5) #

unet_model.train()

for epoch_index in range(num_epochs):
    scheduler.step()
    logging.info("Epoch: {}".format(epoch_index + 1))
    logging.info("Current Learning Rate: {}".format(scheduler.get_lr()))
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
        # print(jmatrix_label.size()) [16, 4, 17, 17]

        csi_data = Variable(csi_data.cuda())
        xy = Variable(jmatrix_label[:, 0:2, :, :].cuda())
        confidence = Variable(jmatrix_label[:, 2:4, :, :].cuda()) # [c, c, y, x] 
        # print(confidence[0].cpu().numpy())

        pred_xy = unet_model(csi_data)

        loss = criterion_L2(torch.mul(confidence, pred_xy), torch.mul(confidence, xy))
        # loss = criterion_L2(pred_xy, xy)

        logging.info("Batch [{} / {}], Loss: {:.6f}".format(batch_index + 1, batch_num, loss.item()))
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    torch.cuda.empty_cache()

    endl = time.time()
    logging.info('Epoch Time: {} minutes'.format((endl - start) / 60))

torch.save(unet_model.state_dict(), '/home/featurize/lsy/WPCP/weights/unet_model_changeshape.pkl')  # 保存 U-Net 模型的参数
# summary(unet_model, (3, 256, 256), batch_size=1)
logging.info("Training finished.")