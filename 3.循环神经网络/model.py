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
from utils import lib
import os
import numpy as np

'''
    模型优化方法：
        1.添加一个新的全连接层作为输出层，激活函数处理
        2.把双向的lstm的output穿一个单向lstm再进行处理
'''


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
        # 加入双向LSTM
        '''
            使用双向LSTM至少需要两层的num_layers
        '''
        self.lstm_fb = nn.LSTM(input_size=100, hidden_size=lib.hidden_size,
                            num_layers=lib.num_layer, batch_first=True,
                            bidirectional=lib.bidriectional, dropout=lib.dropout)
        # 加入单向LSTM
        self.lstm = nn.LSTM(input_size=100, hidden_size=lib.hidden_size,
                            num_layers=lib.num_layer, batch_first=True,
                            dropout=lib.dropout)
        self.fc1 = nn.Linear(lib.hidden_size*2, lib.max_len*100)
        self.fc2 = nn.Linear(lib.max_len*lib.hidden_size, 100)
        self.fc3 = nn.Linear(100, 2)


    def forward(self, input):
        '''
        :param input: input的形状[batch_size, max_len], max_len当前 输入到模型的句子 的最大长度(即句子中词的个数)
        :return:
        '''
        x = self.embedding(input)  # 进行embedding操作，形状变为: [batch_size, max_len, 100]
        x, (h_n, c_n) = self.lstm_fb(x)
        output_fw = h_n[-2, :, :]  # 正向最后一次输出
        output_bw = h_n[-1, :, :]  # 反向最后一次输出
        output = torch.cat([output_fw, output_bw], dim=-1)  # 将正反向的结果进行拼接,得到最终结果， [batch_size, hidden_size*2]

        x = self.fc1(output)   # [batch_size, max_len]  # 使用全连接层改变形状
        x = F.relu(x)   # 使用激活函数不会改变形状

        x = x.view([-1, lib.max_len, 100])  # [batch_size, max_len, 100]
        x, (h_n, c_n) = self.lstm(x)   # x: [batch_size, max_len, hidden_size]
        x = x.contiguous().view([-1, lib.max_len*lib.hidden_size])
        x = self.fc2(x)   # [batch_size, 100]
        x = F.relu(x)
        x = self.fc3(x)  # [batch_size, 2]
        return F.log_softmax(x, dim=-1)  # 因为 output已经经过变形，batch_size在第一维，故不需要再进行变形


'''
    在GPU上运行，只需要将模型(model)和输入、标签(input, label)转换为在cuda上运行即可
'''
imdbmodel = ImdbModel().to(lib.device)
optimizer = optim.Adam(imdbmodel.parameters(), lr=1e-3)

if os.path.exists('./model/model.pkl'):
    '''
        导入模型是 模型的结果必须是实例化之后的那个变量 imdbmodel.load_state_dict 即 imdbmodel
        
        加载模型时 不需要再对实例赋一下值
        下面的写法是错误的
        imdbmodel = imdbmodel.load_state_dict(torch.load('./model/model.pkl'))
    '''
    imdbmodel.load_state_dict(torch.load('./model/model.pkl'))

def train(epoch):
    for index, (input, label) in tqdm(enumerate(get_dataloader(train=True, batch_size=100))):
        input = input.to(lib.device)
        label = label.to(lib.device)

        # 梯度归0
        optimizer.zero_grad()
        output = imdbmodel(input)

        # print('output, label', output, label)
        loss = F.nll_loss(output, label)
        loss.backward()
        optimizer.step()
        print(epoch, index, loss.data)

        if index%100 == 0:
            torch.save(imdbmodel.state_dict(), './model/model.pkl')
            torch.save(optimizer.state_dict(), './model/optimizer.pkl')


def test():
    loss_list = []
    acc_list = []
    data_loader = get_dataloader(train=False, batch_size=200)
    for index, (input, label) in tqdm(enumerate(data_loader), total=len(data_loader), ascii=True, desc='测试：'):
        input = input.to(lib.device)
        label = label.to(lib.device)

        with torch.no_grad():
            output = imdbmodel(input)
            cur_loss = F.nll_loss(output, label)
            loss_list.append(cur_loss.cpu().item())  # 因为是在GPU上训练，取数时需要先将其转化为cpu，才能取数
            # 计算准确率
            '''
                tensor.max(dim=-1) 表示在行这一维度上取tensor的最大值，返回值为：最大值，对应索引位置
            '''
            pred = output.max(dim=-1)[-1]  # 取出预测结果中概率最大对应的索引值
            '''
                判断预测值是否和标签相等 pred.eq(label) 得到布尔值
                .float() 将布尔值转换为float浮点型
                .mean()  对整个tensor求平均值
            '''
            cur_acc = pred.eq(label).float().mean()  # 获取这一个
            acc_list.append(cur_acc.cpu().item())  # 因为是在GPU上训练，取数时需要先将其转化为cpu，才能取数

    print('\ntotal loss:({})/acc:({})'.format(np.mean(loss_list), np.mean(acc_list)))


if __name__ == '__main__':
    for epoch in range(3):
        train(epoch)

    test()
