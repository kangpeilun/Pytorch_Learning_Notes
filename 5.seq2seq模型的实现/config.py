# -*- coding: utf-8 -*-
# date: 2021/10/23
# Project: Pytorch学习
# File Name: config.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com

from data.num_sequence import Num_sequence

num_sequence = Num_sequence()


# 模型配置
train_batch_size = 128  # 训练
test_batch_size = 256   # 测试
max_len = 9  # 句子的最大长度
embedding_dim = 100
# GRU
num_layer = 1
hidden_size = 64