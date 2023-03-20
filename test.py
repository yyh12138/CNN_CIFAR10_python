import torch
import torchvision.transforms
from PIL import Image
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential


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

image = Image.open("./image/dog4.png")
model = torch.load("./myCNN_last.pth")
image = image.convert("RGB")
transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()]
)
image = transform(image)

print(model)
image = torch.reshape(image, (1, 3, 32, 32))
image = image.cuda()
model.eval()
with torch.no_grad():
    output = model(image)

idx = output.argmax(1).item()

test_data = torchvision.datasets.CIFAR10("./torchvision_dataset", train=False,
                                         transform=torchvision.transforms.ToTensor(), download=True)

for k, v in test_data.class_to_idx.items():
    if v == idx:
        print(k)
