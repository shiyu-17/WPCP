import torch.nn as nn
from torchsummary import summary

#定义网络结构
net = nn.Sequential(
            nn.Conv2d(1,8,kernel_size=7),
            nn.MaxPool2d(2,stride=2),
            nn.ReLU(True),
            nn.Conv2d(8,10,kernel_size=5),
            nn.MaxPool2d(2,stride=2),
            nn.ReLU(True)
        )

#输出每层网络参数信息
summary(net,(1,28,28),batch_size=1,device="cpu")


