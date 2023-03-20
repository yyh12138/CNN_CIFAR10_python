from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
# import numpy as np
# from PIL import Image
# import os
import torchvision
import torch

# class MyData(Dataset):
#     def __init__(self, root_dir, label_dir):
#         self.root_dir = root_dir
#         self.label_dir = label_dir
#         self.path = os.path.join(self.root_dir, self.label_dir)
#         self.img_path = os.listdir(self.path)
#     def __getitem__(self, idx):
#         img_name = self.img_path[idx]
#         img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
#         img = Image.open(img_item_path)
#         label = self.label_dir
#         return img, label
#     def __len__(self):
#         return len(self.img_path)

# ants_label_dir = "ants"
# root_dir = "data/train"
# ants_dataset = MyData(root_dir, ants_label_dir)

# writer = SummaryWriter("logs")
# for i in range(100):
#     writer.add_scalar("y=x", i, i)
# writer.close()

# writer = SummaryWriter("logs")
# img_path = "data/train/ants/0013035.jpg"
# img_PIL = Image.open(img_path)
# img_array = np.array(img_PIL)
# writer.add_image("test", img_array, 1, dataformats="HWC")

# # 定义一个处理图片的模板函数
# dataset_transform = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor(),
# ])
# # 6万张32*32的彩色图片集，共10类，5万为训练图片，1万为测试图片
# # train_set = torchvision.datasets.CIFAR10("./torchvision_dataset", train=True, transform=dataset_transform, download=True)
# test_set = torchvision.datasets.CIFAR10("./torchvision_dataset", train=False, transform=dataset_transform, download=True)
#
# # 一次随机取64张图片，共157张大图
# test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
# imgs, targets = test_set[0]
# # 展示数据
# step = 0
# writer = SummaryWriter("torchvision_logs")
# for data in test_loader:
#     imgs, targets = data
#     writer.add_images("test_set", imgs, step)
#     step = step + 1
# writer.close()

#
# input = torch.tensor(
#     [2, 1, 5]
# )
# input = torch.reshape(input, (1, -1, 1, 1))
# print(torch.cuda.is_available())

target = torch.tensor([0, 1])
preds = torch.tensor([0, 0])
print((preds == target).sum().item())
