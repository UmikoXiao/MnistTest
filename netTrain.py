import os.path
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt

from torch.nn import CrossEntropyLoss
from torch import optim
from loadData import DealDataset

save_path = "./LeNet5_re1.pth"  # 模型权重参数保存位置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 创建GPU运算环境
print(device)

BATCH_SIZE = 512  # 超参数,每步训练batch大小,防止过拟合
EPOCH = 50  # 总共训练轮数
learning_rate = 0.05  # 学习率
classNum=10

"""采用的网络框架"""
from model.MLPModel import MLP
from model.leNetModel import LeNet1, LeNet4, LeNet5
from model.resNetModel import ResNet18, ResNet34, ResNet50

testFramwork = ResNet18

"""采用的优化器"""
opt = optim.SGD
# opt = optim.Adam

"""采用的损失函数"""
lossFunc = CrossEntropyLoss()  # 交叉熵损失函数


def testTurn(model, loader):
    correct = 0
    # validation和test过程不需要反向传播
    model.eval()
    # 取测试数据
    y_m,eta_m=[],[]
    for X, y in loader:
        X, y = X.to(device), y.to(device)

        with torch.no_grad():
            outputs = model(X)  # 计算测试数据的输出
            # 计算出out在第一维度上最大值对应编号，得模型的预测值
            _, eta = torch.max(outputs.data, dim=1)
        # 是否正确？
        correct += torch.eq(eta, y).float().sum().item()
        y_m = np.append(y_m, y.cpu())
        eta_m=np.append(eta_m,eta.cpu())

    return correct / len(loader.dataset),f1_score(y_m,eta_m,average='micro')


def trainTurn(model, loader, optimizer, epoch):
    running_loss = 0.0  # 初始化loss
    running_total = 0
    running_correct = 0
    for batch_idx, data in enumerate(loader, 0):
        X, y = data
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        # forward + backward + update
        outputs = model(X)
        loss = lossFunc(outputs, y)
        loss.backward()
        optimizer.step()

        # 统计loss
        running_loss += loss.item()
        # 计算正确率
        _, eta = torch.max(outputs.data, dim=1)
        running_total += X.shape[0]
        running_correct += torch.eq(eta, y).float().sum().item()
        print('\r[epoch %d,batch %5d]: loss: %.3f , acc: %.2f %%'
              % (epoch + 1, batch_idx + 1, running_loss / 300, 100 * running_correct / running_total), end='')
    return running_correct / running_total


def eval(X,model,checkPointPath):
    # 在新输入上应用模型
    if not isinstance(model, torch.nn.Module):
        model = model(classNum)
    model.load_state_dict(torch.load(checkPointPath))
    model.eval()
    return [model(x) for x in X]

def main():
    import time

    t0 = time.time()
    # 加载数据集，指定训练或测试数据，指定于处理方式
    trainDataset = DealDataset(os.path.abspath(r'.\dataset'), "train-images-idx3-ubyte.gz",
                               "train-labels-idx1-ubyte.gz",
                               transform=transforms.ToTensor())
    testDataset = DealDataset(os.path.abspath(r'.\dataset'), "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz",
                              transform=transforms.ToTensor())
    train_dataloader = DataLoader(trainDataset, BATCH_SIZE, True, num_workers=0)
    test_dataloader = DataLoader(testDataset, BATCH_SIZE, True, num_workers=0)

    net = testFramwork(classNum)
    net.to(device)  # 实例化网络模型并送入GPU
    # net.load_state_dict(torch.load(save_path))  # 使用上次训练权重接着训练
    optimizer = opt(net.parameters(), learning_rate)
    bestAcc = 0

    accListTest, accListTrain,f1 = [], [],[]
    for e in range(EPOCH):
        net.train()
        accTrain = trainTurn(net, train_dataloader, optimizer, e)
        print()
        accTest,f1Test = testTurn(net, test_dataloader)
        if accTest > bestAcc:
            bestAcc = accTest
            # torch.save(net.state_dict(), save_path)
        # else:
        #     net.load_state_dict(torch.load(save_path))
        print('[epoch %d]: train accuracy: %.2f %%, test accuracy: %.2f %%,best: %.2f %%' % (
            e + 1, 100 * accTrain, 100 * accTest, 100 * bestAcc))
        accListTest.append(accTest)
        accListTrain.append(accTrain)
        f1.append(f1Test)

    print(time.time() - t0)
    plt.plot(accListTest, label='testTurn')
    plt.plot(accListTrain, label='trainTurn')
    plt.plot(f1, label='F1Score')
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('acc %')
    plt.savefig('figure/' + save_path + '.png')
    plt.show(block=True)

if __name__ == '__main__':
    main()