import numpy as np
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, classes_num):
        super(MLP, self).__init__()
        self.inputLayer=nn.Sequential(
            nn.Linear(28*28, 56*56),  # 全连接层升采样
            nn.ReLU(inplace=True),
        )
        self.hidderLayer =nn.Sequential(
            nn.Linear(56*56, 28*28),  # 全连接层降采样
            nn.ReLU(inplace=True),
            nn.Linear(28*28, 14*14),  # 全连接层降采样
            nn.ReLU(inplace=True),
        )

        self.outputLayer=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(14*14, classes_num)
        )  # 全连接层获得结果
    def forward(self, x):
        x = x.reshape(x.shape[0]*x.shape[1], -1)   # 将x展平
        x=self.inputLayer(x)
        x=self.hidderLayer(x)
        x=self.outputLayer(x)
        return x

