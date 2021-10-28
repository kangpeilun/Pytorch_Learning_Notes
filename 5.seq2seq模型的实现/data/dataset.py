# -*- coding: utf-8 -*-
# date: 2021/10/23
# Project: Pytorch学习
# File Name: dataset.py
# Description: 准备数字数据集
# Author: Anefuer_kpl
# Email: 374774222@qq.com
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import config

class NumDataset(Dataset):
    def __init__(self, train=True):
        '''
        @param train: 默认为True，表示生成测试集
        '''
        # 使用numpy随机创建一组数据
        np.random.seed(10)  # 设置随机种子，防止每次重新训练之后，数据集发生变化
        self.data = np.random.randint(0, 1e8, size=[500000])
        if train:
            self.data = self.data[:400000]  # 使用前 400000 个数据用于训练
        else:
            self.data = self.data[400000:]  # 使用后 100000 个数据用于测试

    def __getitem__(self, item):
        '''
            传入模型的应该是字符串，而不是数字
            list将字符串转换为列表
        '''
        input = list(str(self.data[item]))
        label = list(str(self.data[item])+'0')
        input_length = len(input)
        label_length = len(label)
        return input, label, input_length, label_length

    def __len__(self):
        return self.data.shape[0]

def collate_fn(batch):
    '''
        zip(*batch) 对每一个batch进行解包
        [(input1, label1), (input2, label2)] --> [(input1, input2), (label1, label2)]

        batch:[(input1, lable1), ...]
        使用编码器编码的时候 需要对句子进行 从长到短的排序，故可以在collate_fn中提前将这一部分完成
        # 根据batch的第三个值的大小进行排序，reverse=True取反表示从大到小
        sorted(batch, key=lambda x:x[2], reverse=True)
    '''
    batch = sorted(batch, key=lambda x:x[2], reverse=True)
    input, label, input_length, label_length = zip(*batch)
    input = torch.LongTensor([config.num_sequence.transform(i, max_len=config.max_len) for i in input])  # 对每一个输入进行处理
    label = torch.LongTensor([config.num_sequence.transform(i, max_len=config.max_len+1, add_eos=True) for i in label])  # 对每一个输入进行处理 [[],[]]
    input_length = torch.LongTensor(input_length)
    label_length = torch.LongTensor(label_length)
    return input, label, input_length, label_length


train_dataloader = DataLoader(NumDataset(train=True), batch_size=config.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
test_dataloader = DataLoader(NumDataset(train=False), batch_size=config.test_batch_size, collate_fn=collate_fn, drop_last=True)

if __name__ == '__main__':
    for input, label, input_length, label_length in train_dataloader:
        print(input, input.size())
        print(label, label.size())
        print(input_length)
        print(label_length)
        break
