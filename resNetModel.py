import torch
import torch.nn as nn
from torch.nn import functional as F


class CommonBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):  # 普通Block
        super(CommonBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False)  # 3大小卷积，输入=输出
        self.bn1 = nn.BatchNorm2d(out_channel)  # 卷积后归一化处理，防止ReLU过大导致性能不稳定
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, stride, 1, bias=False)  # 3大小卷积，输入=输出
        self.bn2 = nn.BatchNorm2d(out_channel)  # 卷积后归一化处理，防止ReLU过大导致性能不稳定

    def forward(self, x):
        identity = x  # 普通Block的shortcut为直连，不需要升维下采样

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)  # 第一次卷积并relu
        x = self.bn2(self.conv2(x))  # 第二次卷积，不relu

        x += identity  # 旁路x相加，维持梯度
        return F.relu(x, inplace=True)  # relu后输出


class SpecialBlock(nn.Module):  # 特殊Block
    def __init__(self, in_channel, out_channel, stride):  # 注意这里的stride传入一个数组，shortcut和残差部分stride不同
        super(SpecialBlock, self).__init__()
        self.ResTransform = nn.Sequential(  # 对x进行升维处理
            nn.Conv2d(in_channel, out_channel, 1, stride[0], 0, bias=False),  # 升维卷积，1大小卷积核
            nn.BatchNorm2d(out_channel)  # 卷积后归一化处理，防止ReLU过大导致性能不稳定
        )
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride[0], 1, bias=False)  # 等维或升维卷积，3大小卷积核
        self.bn1 = nn.BatchNorm2d(out_channel)  # 卷积后归一化处理，防止ReLU过大导致性能不稳定
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, stride[1], 1, bias=False)  # 等维卷积，3大小卷积核
        self.bn2 = nn.BatchNorm2d(out_channel)  # 卷积后归一化处理，防止ReLU过大导致性能不稳定

    def forward(self, x):
        identity = self.ResTransform(x)  # 调用change_channel对输入升维

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)  # 第一次残差卷积后relu
        x = self.bn2(self.conv2(x))  # 第二次残差卷积，不relu

        x += identity  # 旁路x相加，维持梯度
        return F.relu(x, inplace=True)  # 输出卷积单元


class IdentityBlock(nn.Module):  # 步长设置为2时即为conv BLock
    def __init__(self, in_channel, hidden_channel, out_channel, stride=1):  # 3层conv,根据需求决定是否改变结构
        super(IdentityBlock, self).__init__()
        self.upsample = in_channel != out_channel
        
        self.conv1 = nn.Conv2d(in_channel, hidden_channel, 1, 1, 0, bias=False)  # 1大小卷积，输入=输出
        self.bn1 = nn.BatchNorm2d(hidden_channel)  # 卷积后归一化处理，防止ReLU过大导致性能不稳定
        self.conv2 = nn.Conv2d(hidden_channel, hidden_channel, 3, stride, 1, bias=False)  # 3大小卷积，若需降采样，在此处改变步长
        self.bn2 = nn.BatchNorm2d(hidden_channel)  # 卷积后归一化处理，防止ReLU过大导致性能不稳定
        self.conv3 = nn.Conv2d(hidden_channel, out_channel, 1, 1, 0, bias=False)  # 1大小卷积，输入=输出
        self.bn3 = nn.BatchNorm2d(out_channel)  # 卷积后归一化处理，防止ReLU过大导致性能不稳定
        if self.upsample:
            self.ResTransform = nn.Sequential(  # 维度发生改变时，对x进行升维处理
                nn.Conv2d(in_channel, out_channel, 1, stride, 0, bias=False),  # 升维卷积，1大小卷积核，若需降采样，在此处改变步长
                nn.BatchNorm2d(out_channel)  # 卷积后归一化处理，防止ReLU过大导致性能不稳定
            )

    def forward(self, x):
        identity = x
        if self.upsample:
            identity = self.ResTransform(x)  # 对x进行升维

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)  # 第一次卷积relu
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)  # 第二次卷积relu
        x = self.bn3(self.conv3(x))  # 第三次卷积不relu

        x += identity  # 旁路x相加，维持梯度
        return F.relu(x, inplace=True)  # relu后输出


class ResNet18(nn.Module):
    def __init__(self, classes_num):
        super(ResNet18, self).__init__()
        self.prepare = nn.Sequential(  # 所有的ResNet共有的预处理
            nn.Conv2d(1, 64, 7, 2, 3),  # mnist为28*28*1单通道数据集，2步长二维卷积=>64*14*14
            nn.BatchNorm2d(64),  # 卷积后归一化处理，防止ReLU过大导致性能不稳定
            nn.ReLU(inplace=True),  # ReLU激活
            nn.MaxPool2d(3, 2, 1)  # 2步长最大池化，维持矩阵大小=>64*7*7
        )
        self.layer1 = nn.Sequential(  # 两个CommonBlock四次卷积，64*7*7=>64*7*7
            CommonBlock(64, 64, 1),
            CommonBlock(64, 64, 1)
        )
        self.layer2 = nn.Sequential(  # 升维，Special+Common,64*7*7=>128*4*4
            SpecialBlock(64, 128, [2, 1]),
            CommonBlock(128, 128, 1)  # CommonBlock两次卷积，128*4*4=>128*4*4
        )
        self.layer3 = nn.Sequential(
            SpecialBlock(128, 256, [2, 1]),  # 升维下采样，Special+Common,128*4*4=>256*2*2
            CommonBlock(256, 256, 1)  # CommonBlock两次卷积，256*2*2=>256*2*2
        )
        self.layer4 = nn.Sequential(
            SpecialBlock(256, 512, [2, 1]),  # 升维下采样，Special+Common,256*2*2=>512*1*1
            CommonBlock(512, 512, 1)  # CommonBlock两次卷积，512*1*1=>512*1*1
        )
        # self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))    # 卷积结束，通过一个自适应均值池化 512*1*1=>512*1*1 *其实是不需要池化的
        self.fc = nn.Sequential(  # 最后用于分类的全连接层
            nn.Dropout(p=0.5),  # 训练阶段Dropout防止过拟合，0.5概率
            nn.Linear(512, classes_num)  # mnist10分类问题，全连接层降维10
        )

    def forward(self, x):
        x = self.prepare(x)  # 预处理
        x = self.layer1(x)  # 四个卷积单元
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.pool(x)            # 自适应均值池化
        x = x.reshape(x.shape[0], -1)  # 将x展平
        x = self.fc(x)  # 全连接层到10分类

        return x


class ResNet34(nn.Module):
    def __init__(self, classes_num):
        super(ResNet34, self).__init__()
        self.prepare = nn.Sequential(  # 所有的ResNet共有的预处理
            nn.Conv2d(1, 64, 7, 2, 3),  # mnist为28*28*1单通道数据集，2步长二维卷积=>64*14*14
            nn.BatchNorm2d(64),  # 卷积后归一化处理，防止ReLU过大导致性能不稳定
            nn.ReLU(inplace=True),  # ReLU激活
            nn.MaxPool2d(3, 2, 1)  # 2步长最大池化，维持矩阵大小=>64*7*7
        )
        self.layer1 = nn.Sequential(  # 3个CommonBlock四次卷积，64*7*7=>64*7*7
            CommonBlock(64, 64, 1),
            CommonBlock(64, 64, 1),
            CommonBlock(64, 64, 1)
        )
        self.layer2 = nn.Sequential(  # 升维，Special+Common*3,64*7*7=>128*4*4
            SpecialBlock(64, 128, [2, 1]),
            CommonBlock(128, 128, 1),
            CommonBlock(128, 128, 1),
            CommonBlock(128, 128, 1)
        )
        self.layer3 = nn.Sequential(
            SpecialBlock(128, 256, [2, 1]),  # 升维下采样，Special+Common*5,128*4*4=>256*2*2
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1),
        )
        self.layer4 = nn.Sequential(
            SpecialBlock(256, 512, [2, 1]),  # 升维下采样，Special+Common*3,256*2*2=>512*1*1
            CommonBlock(512, 512, 1),
            CommonBlock(512, 512, 1)
        )
        # self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))    # 卷积结束，通过一个自适应均值池化 512*1*1=>512*1*1 *其实是不需要池化的
        self.fc = nn.Sequential(  # 最后用于分类的全连接层
            nn.Dropout(p=0.5),  # 训练阶段Dropout防止过拟合，0.5概率
            nn.Linear(512, classes_num)  # mnist10分类问题，全连接层降维10
        )

    def forward(self, x):
        x = self.prepare(x)  # 预处理
        x = self.layer1(x)  # 四个卷积单元
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.pool(x)            # 自适应均值池化
        x = x.reshape(x.shape[0], -1)  # 将x展平
        x = self.fc(x)  # 全连接层到10分类

        return x


class ResNet50(nn.Module):
    def __init__(self, classes_num):
        super(ResNet50, self).__init__()
        self.prepare = nn.Sequential(  # 所有的ResNet共有的预处理
            nn.Conv2d(1, 64, 7, 2, 3),  # mnist为28*28*1单通道数据集，2步长二维卷积=>64*14*14
            nn.BatchNorm2d(64),  # 卷积后归一化处理，防止ReLU过大导致性能不稳定
            nn.ReLU(inplace=True),  # ReLU激活
            nn.MaxPool2d(3, 2, 1)  # 2步长最大池化，维持矩阵大小=>64*7*7
        )
        self.layer1 = nn.Sequential(  # 3个Identity但不重采样，64*7*7=>256*7*7
            IdentityBlock(64, 64, 256, 1),
            IdentityBlock(256, 64, 256, 1),
            IdentityBlock(256, 64, 256, 1)
        )
        self.layer2 = nn.Sequential(  # 4个Identity升维降采样，64*7*7=>128*4*4
            IdentityBlock(256, 128, 512, 2),
            IdentityBlock(512, 128, 512, 1),
            IdentityBlock(512, 128, 512, 1),
            IdentityBlock(512, 128, 512, 1),
        )
        self.layer3 = nn.Sequential(  # 6个Identity升维降采样，128*4*4=>256*2*2
            IdentityBlock(512, 256, 1024, 2),
            IdentityBlock(1024, 256, 1024, 1),
            IdentityBlock(1024, 256, 1024, 1),
            IdentityBlock(1024, 256, 1024, 1),
            IdentityBlock(1024, 256, 1024, 1),
            IdentityBlock(1024, 256, 1024, 1)
        )
        self.layer4 = nn.Sequential(  # 3个Identity升维降采样，256*2*2=>512*1*1
            IdentityBlock(1024, 512, 2048, 2),
            IdentityBlock(2048, 512, 2048, 1),
            IdentityBlock(2048, 512, 2048, 1)
        )
        # self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))    # 卷积结束，通过一个自适应均值池化 512*1*1=>512*1*1 *其实是不需要池化的
        self.fc = nn.Sequential(  # 最后用于分类的全连接层
            nn.Dropout(p=0.5),  # 训练阶段Dropout防止过拟合，0.5概率
            nn.Linear(2048, 512),  # 维度太高了，先下降一下
            nn.ReLU(inplace=True),
            nn.Linear(512, classes_num)  # mnist10分类问题，全连接层降维10
        )

    def forward(self, x):
        x = self.prepare(x)  # 预处理
        x = self.layer1(x)  # 四个卷积单元
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.pool(x)            # 自适应均值池化
        x = x.reshape(x.shape[0], -1)  # 将x展平
        x = self.fc(x)  # 全连接层到10分类

        return x
