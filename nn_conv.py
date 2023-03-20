import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./torchvision_dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset=dataset, batch_size=64)

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2, stride=1),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2, stride=1),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2, stride=1),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

myCNN = MyCNN()

loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(myCNN.parameters(), lr=0.005)

for epoch in range(20):
    # 每轮的预测误差和
    total_loss = 0.0
    # 一轮学习
    for data in dataloader:
        imgs, targets = data
        outputs = myCNN(imgs)
        result_loss = loss(outputs, targets)
        # 梯度清零
        optim.zero_grad()
        result_loss.backward()
        # 权重参数更新
        optim.step()
        total_loss += result_loss
    print(total_loss)