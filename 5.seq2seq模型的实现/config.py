# -*- coding: utf-8 -*-
# date: 2021/10/23
# Project: Pytorch学习
# File Name: config.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com

import torch
from data.num_sequence import Num_sequence

num_sequence = Num_sequence()

# 根据cuda是否可用，自动选择训练的设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 模型保存
model_save_path = 'model/seq2seq.pt'
optimizer_save_path = 'model/optimizer.pt'

# 模型配置
train_batch_size = 512  # 训练
test_batch_size = 256   # 测试
max_len = 9  # 句子的最大长度
num_embedding = len(num_sequence)  # 词典中不同种类词的个数
embedding_dim = 100     # 词向量的维度
# GRU
num_layer = 1
hidden_size = 64
# decoder
teacher_forcing_ratio = 0.5