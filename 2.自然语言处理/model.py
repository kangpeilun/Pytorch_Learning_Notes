# -*- coding:utf-8 -*-
# author: Anefuer_kpl
# Email: 374774222@qq.com
# datatime: 2021-09-25 13:58
# project: Pytorch学习

import torch
from torch import nn
import torch.nn.functional as F
from model_data.build_dataset import ws, get_dataloader, MAX_LEN
from torch import optim
from tqdm import tqdm


# 定义模型
class ImdbModel(nn.Module):
    def __init__(self):
        super(ImdbModel, self).__init__()
        '''
        nn.Embedding传入两个参数：
            num_embeddings： 整个数据集中不同词的个数
            embedding_dim： 词向量的维度
        '''
        self.embedding = nn.Embedding(len(ws), 100)
        self.fc1 = nn.Linear(MAX_LEN * 100, 2)

    def forward(self, input):
        '''
        :param input: input的形状[batch_size, max_len], max_len当前 输入到模型的句子 的最大长度(即句子中词的个数)
        :return:
        '''
        x = self.embedding(input)  # 进行embedding操作，形状变为: [batch_size, max_len, 100]
        x = x.view([-1, MAX_LEN * 100])
        out = self.fc1(x)

        return F.log_softmax(out)


imdbmodel = ImdbModel()
optimizer = optim.Adam(imdbmodel.parameters(), lr=1e-3)
dataloader = get_dataloader()


def train(epoch):
    for index, (input, label) in tqdm(enumerate(dataloader)):
        # 梯度归0
        optimizer.zero_grad()
        output = imdbmodel(input)
        print('output, label', output, label)
        loss = F.nll_loss(output, label)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    train(1)
