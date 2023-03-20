import torch.optim
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import *


# 准备数据集
train_data = torchvision.datasets.CIFAR10("./torchvision_dataset", train=True,
                                          transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10("./torchvision_dataset", train=False,
                                         transform=torchvision.transforms.ToTensor(), download=True)
print("训练数据集长度为 {}".format(len(train_data)))
print("测试数据集长度为 {}".format(len(test_data)))
# 加载数据集
train_dataLoader = DataLoader(train_data, batch_size=64)
test_dataLoader = DataLoader(test_data, batch_size=64)
# 创建网络模型
myCNN = MyCNN()
myCNN = myCNN.cuda()
# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
# 优化器
optim = torch.optim.SGD(myCNN.parameters(), 0.01)
# 相关参数
total_train = 0
total_test = 0
# 添加tensorboard
writer = SummaryWriter("./myCNN_logs")

for i in range(10):
    print("第{}轮训练开始".format(i+1))
    myCNN.train()
    for data in train_dataLoader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        output = myCNN(imgs)
        loss = loss_fn(output, targets)
        # 优化器调用
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_train += 1
        if total_train % 100 == 0:
            print("训练次数 {}, loss {}".format(total_train, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train)
    # 训练完一轮后使用测试数据集评估性能
    myCNN.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataLoader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            output = myCNN(imgs)
            loss = loss_fn(output, targets)
            total_test_loss += loss.item()
            # (tensor[0,1] == tensor[1,1]).sum() => tensor(1)
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print("测试集loss {}".format(total_test_loss))
    print("测试集正确率 {}".format(total_accuracy/len(test_data)))
    writer.add_scalar("test_loss", total_test_loss, total_test)
    writer.add_scalar("test_accuracy", total_accuracy/len(test_data), total_test)
    total_test += 1

torch.save(myCNN, "myCNN_last.pth")
writer.close()
