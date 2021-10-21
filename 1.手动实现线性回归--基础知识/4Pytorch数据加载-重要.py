# -*- coding:utf-8 -*-
# author: Anefuer_kpl
# Email: 374774222@qq.com
# datatime: 2021-09-22 12:20
# project: Pytorch学习

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

data_path = r'./data/smsspamcollection/SMSSpamCollection'

# 构建数据集类
class MyDataset(Dataset):
    def __init__(self):
        self.lines = open(data_path, 'r', encoding='utf-8').readlines()

    def __getitem__(self, item):
        # 获取索引对应位置的一条数据, item即为索引
        line = self.lines[item].strip()  # 获取索引对应的一行数据
        label = line[:4].strip() # 获取数据的标签
        content = line[4:].strip() # 获取数据的内容
        return label, content

    def __len__(self):
        # 返回数据的总数量
        return len(self.lines)


if __name__ == '__main__':
    my_dataset = MyDataset()
    # print(my_dataset[0])
    print(len(my_dataset))  # len(dataset) = 数据集的样本数

    data_loader = DataLoader(dataset=my_dataset, batch_size=10, shuffle=True, drop_last=True)  # drop_last=True 当数据集/batch_size 除不下时会将最后一个batch丢掉，防止程序报错
    print(len(data_loader))  # len(dataloader) = math.ceil(样本数/batch_size) 即向上取整
    for index, (label, content) in enumerate(data_loader):
        print(index, label, content)