#! /usr/bin/env python3


import argparse
import copy
import logging
import random
import time

import numpy as np
import torch
import torch.nn as nn
from models import alexnet, densenet, googlenet, mobilenetv2, resnet3, vgg
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

rand_seed = 218276150
torch.manual_seed(rand_seed)
torch.cuda.manual_seed(rand_seed)
torch.cuda.manual_seed_all(rand_seed)
np.random.seed(rand_seed)
random.seed(rand_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class Logger(object):
    def __init__(self, job_name, file_path, log_level=logging.INFO, mode="w"):
        self.__logger = logging.getLogger(job_name)
        self.__logger.setLevel(log_level)
        self.__fh = logging.FileHandler(filename=file_path, mode=mode)
        self.__formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
        self.__fh.setFormatter(self.__formatter)
        self.__logger.addHandler(self.__fh)

    @property
    def logger(self):
        return self.__logger


logger = Logger(job_name="ckpt", file_path="./logs/ckpt.log").logger

model = vgg.VGG16()
for param in model.parameters():
    param.grad = torch.zeros_like(param)

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)


transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
training_dataset = datasets.CIFAR10(root="./data/", train=True, download=True, transform=transform)
data_loader = DataLoader(training_dataset, batch_size=128, shuffle=True)

criterion = nn.CrossEntropyLoss()


for epoch_idx in range(100):
    for iter_idx, (data, target) in enumerate(data_loader):
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        logger.info("sleep_start")
        time.sleep(2)
        logger.info("update_start")
        optimizer.step()
        optimizer.zero_grad()
        logger.info("update_end")
        time.sleep(2)
        logger.info("sleep_end")
