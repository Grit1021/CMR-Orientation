#import packages
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
from model.densenet import DenseNet
from tqdm import tqdm
from util.dataloader import LoadDataset
from d2l import torch as d2l

#记录训练结果
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

#读取文件所在路径
