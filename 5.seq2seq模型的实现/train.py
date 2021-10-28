# -*- coding: utf-8 -*-
# date: 2021/10/25
# Project: Pytorch学习
# File Name: train.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com

import torch
from torch.optim import Adam
import torch.nn.functional as F

from seq2seq.seq2seq_model import Seq2Seq
from data.dataset import train_dataloader, NumDataset

import config
from tqdm import tqdm

# 训练流程
# 1.实例化model，optimizer， loss
# 2.遍历dataloader
# 3.调用模型，得到结果
# 4.计算损失
# 5.模型保存和加载

seq2seq = Seq2Seq().to(config.device)  # 将模型分配给指定设备
optimizer = Adam(seq2seq.parameters(), lr=1e-3)


def train(epoch):
    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), ascii=True)
    for index, (input, label, input_length, label_length) in bar:
        # 将训练数据分配给指定设备
        input = input.to(config.device)
        label = label.to(config.device)
        input_length = input_length.to(config.device)

        optimizer.zero_grad()  # 梯度置0
        decoder_outputs, _ = seq2seq(input, label, input_length, label_length)
        decoder_outputs = decoder_outputs.view(decoder_outputs.size(0) * decoder_outputs.size(1), -1)  # [batch_size*max_len, vacab_size]
        label = label.view(-1)  # [batch_size*max_len]
        loss = F.nll_loss(decoder_outputs, label)  # 计算loss
        loss.backward()  # 反向传播
        optimizer.step()  # 参数更新

        bar.set_description('train epoch:{}\tindex:{}\tloss:{:.3f}'.format(epoch, index, loss.item()))
        if index % 100 == 0:
            torch.save(seq2seq.state_dict(), config.model_save_path)
            torch.save(optimizer.state_dict(), config.optimizer_save_path)



if __name__ == '__main__':
    for i in range(10):
        train(i)
