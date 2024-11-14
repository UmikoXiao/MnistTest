import numpy as np
import torch
import torch.nn as nn

class LeNet1(nn.Module):
    def __init__(self, classes_num):
        super(LeNet1, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(1,4,kernel_size=5,stride=1,padding=0),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.layer2 =nn.Sequential(
            nn.Conv2d(4,12,kernel_size=5,stride=1,padding=0),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.linearLayer=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(192, classes_num)
        )  # 全连接层获得结果
    def forward(self, x):
        x=self.layer1(x)
        x=self.layer2(x)
        x = x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])   # 将x展平
        x=self.linearLayer(x)
        return x

class LeNet4(nn.Module):
    def __init__(self, classes_num):
        super(LeNet4, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(1,4,kernel_size=5,stride=1,padding=0),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.layer2 =nn.Sequential(
            nn.Conv2d(4,12,kernel_size=5,stride=1,padding=0),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.linearLayer=nn.Sequential(
            nn.Linear(192, 120),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(120, classes_num)
        )  # 全连接层获得结果
    def forward(self, x):
        x=self.layer1(x)
        x=self.layer2(x)
        x = x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])   # 将x展平
        x=self.linearLayer(x)
        return x

class LeNet5(nn.Module):
    def __init__(self, classes_num):
        super(LeNet5, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(1,6,kernel_size=5,stride=1,padding=0),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.layer2 =nn.Sequential(
            nn.Conv2d(6,16,kernel_size=5,stride=1,padding=0),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.linearLayer=nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(84, classes_num)
        )  # 全连接层获得结果
    def forward(self, x):
        x=self.layer1(x)
        x=self.layer2(x)
        x = x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])   # 将x展平
        x=self.linearLayer(x)
        return x