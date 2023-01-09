"""
    Orientation Recognition Network training
    transfer training on LGE dataset

"""

import torch
import torchvision
from torchvision import transforms
from torch import nn
import torch.utils.data
import sys

import cv2
import scipy
import glob
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from data_processing.dataloader import LoadDataset
sys.path.append('/home/liyuxin/Orientation-Adjust-Tool-master/MSCMR_orient-master/code/d2l')
from d2l import torch as d2l

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

os.environ['CUDA_VISIBLE_DEVICES']='2'
root = 'data_transform/LGE'  # 数据路径
num_workers=8
PATH = './checkpoints/C0/model-best.pth'  # 加载模型路径(来自pre-training所获得的C0数据集)
# 固定网络参数，learning_rate和batch_size可以设置大一点
Init_Epoch          = 0
Fix_Epoch        = 20
Fix_batch_size   = 32
Fix_lr           = 1e-3
# 迁移训练参数，learning_rate和batch_size设置小一点
Free_Epoch      = 40
Free_batch_size = 16
Free_lr         = 1e-4

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
makedir('./checkpoints/LGE') #读入数据路径

#proposed network
net = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(32, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(64 * 16 * 16, 64), nn.Sigmoid(),
    nn.Linear(64, 8)
)

# #operational DNN
# net = nn.Sequential(
#     nn.Conv2d(3, 32, kernel_size=3, padding=1),
#     nn.BatchNorm2d(32),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=4, stride=4),
#     nn.Conv2d(32, 32, kernel_size=3, padding=1),
#     nn.BatchNorm2d(32),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=4, stride=4),
#     nn.Conv2d(32, 32, kernel_size=3, padding=1),
#     nn.BatchNorm2d(32),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2, stride=2),
#     nn.AvgPool2d(kernel_size=2, stride=2),
#     nn.Flatten(),
#     nn.Linear(32* 4 * 4, 32), nn.Sigmoid(),
#     nn.Linear(32, 8)
# )


# #ResNet18 alternative network
# from torchvision.models import resnet18
# net = resnet18() 
# num_ftrs = net.fc.in_features
# net.fc = nn.Linear(num_ftrs, 8)


net.load_state_dict(torch.load(PATH))

device = d2l.try_gpu(2)
print('training on', device)
net.to(device)
loss = nn.CrossEntropyLoss()


'''
Fix the network backbone
training
'''
Fix_Train        = True
batch_size  = Fix_batch_size
lr          = Fix_lr
start_epoch = Init_Epoch
end_epoch   = Fix_Epoch

dataset=LoadDataset(root=root, mode='train', truncation = True)
train_iter=DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers)
dataset=LoadDataset(root=root, mode='valid', truncation = True)
valid_iter=DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
dataset=LoadDataset(root=root, mode='test', truncation = True)
test_iter=DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=num_workers)
timer, num_batches = d2l.Timer(), len(train_iter)
if Fix_Train:
    # 全连接层之前的参数全部固定
    # for param in net[:12].parameters():
    for param in net[:9].parameters():
        param.requires_grad = False
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

best_acc = 0
valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
writer.add_scalar('valid acc', valid_acc, 0)
for epoch in range(start_epoch, end_epoch):
    print(f'epoch:{epoch}')
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = d2l.Accumulator(3)
    net.train()
    for i, (X, y) in tqdm(enumerate(train_iter)):
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()
        with torch.no_grad():
            metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
            writer.add_scalar('train loss', train_l, epoch + (i + 1) / num_batches)
            writer.add_scalar('train acc', train_acc, epoch + (i + 1) / num_batches)
    valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
    writer.add_scalar('valid acc', valid_acc, epoch + 1)

    if valid_acc>best_acc:
        torch.save(net.state_dict(), "./checkpoints/LGE/model-best.pth")
        best_acc = valid_acc
    torch.save(net.state_dict(), "./checkpoints/LGE/model-latest.pth")



"""finetune："""
# 迁移训练
Freeze_Train        =False
batch_size  = Free_batch_size
lr          = Free_lr
start_epoch = Fix_Epoch
end_epoch   = Free_Epoch

dataset=LoadDataset(root=root, mode='train', truncation = True)
train_iter=DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers)
dataset=LoadDataset(root=root, mode='valid', truncation = True)
valid_iter=DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
dataset=LoadDataset(root=root, mode='test', truncation = True)
test_iter=DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=num_workers)
timer, num_batches = d2l.Timer(), len(train_iter)
if not Fix_Train:
    # for param in net[:12].parameters():
    for param in net[:9].parameters():
        param.requires_grad = True
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

for epoch in range(start_epoch,end_epoch):
    print(f'epoch:{epoch}')
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = d2l.Accumulator(3)
    net.train()
    for i, (X, y) in tqdm(enumerate(train_iter)):
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()
        with torch.no_grad():
            metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
            writer.add_scalar('train loss', train_l, epoch + (i + 1) / num_batches)
            writer.add_scalar('train acc', train_acc, epoch + (i + 1) / num_batches)
    valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
    writer.add_scalar('valid acc', valid_acc, epoch + 1)

    if valid_acc>best_acc:
        torch.save(net.state_dict(), "./checkpoints/LGE/model-best.pth")
        best_acc = valid_acc
    torch.save(net.state_dict(), "./checkpoints/LGE/model-latest.pth")

num_epochs=40
print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
      f'test acc {valid_acc:.3f}')
print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
      f'on {str(device)}')