#-*- coding:utf-8 -*-
# author: Anefuer_kpl
# Email: 374774222@qq.com
# datatime: 2021-10-06 15:18
# project: Pytorch学习

from torch import nn
import torch

batch_size = 10
seq_len = 20 # 句子长度
vocab_size = 100 # 词典的数量
embedding_dim = 30 # 用长度为30的向量表示一个词语
hidden_size = 18  # lstm单元个数
num_layers = 1 # lstm层的个数

# 构造一个batch的数据
input = torch.randint(low=0, high=100, size=[batch_size, seq_len])  #[10, 20]
h_0 = torch.randn(size=[num_layers*1, batch_size, hidden_size])  #[1, 10, 18]
c_0 = torch.randn(size=[num_layers*1, batch_size, hidden_size])  #[1, 10, 18]

# 数据经过embedding处理
'''
    nn.Embedding传入两个参数：
        num_embedding: 词的个数
        embedding_dim: 词向量的维度
'''
embedding = nn.Embedding(vocab_size, embedding_dim)
input_embeded = embedding(input) # batch seq_len embedding_dim [10, 20, 30]

# 把embedding后的数据传入lstm
lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
output, (h_n, c_n) = lstm(input_embeded, (h_0, c_0)) # output:[10, 20, 18]    h_n:[1*1, 10, 18]    c_n:[1*1, 10, 18]

print(output)
print('======================')
print(h_n)
print('======================')
print(c_n)