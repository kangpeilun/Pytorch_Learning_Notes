# -*- coding: utf-8 -*-
# date: 2021/10/25
# Project: Pytorch学习
# File Name: encoder_model.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import config


class NumEncoder(nn.Module):
    def __init__(self):
        super(NumEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=config.num_embedding,
                                      embedding_dim=config.embedding_dim,
                                      padding_idx=config.num_sequence.PAD)  # 指定 PAD 不需要进行更新，这里需要传入的是数值
        self.gru = nn.GRU(input_size=config.embedding_dim,
                          hidden_size=config.hidden_size,
                          num_layers=config.num_layer,
                          batch_first=True,)

    def forward(self, input, input_length):
        '''
        @param input: [batch_size, max_len]
        @return:
        '''
        embeded = self.embedding(input)  # [batch_size, max_len, embedding_dim]
        embeded = pack_padded_sequence(embeded, input_length, batch_first=True)  # 对embeded结果进行打包，加速更新速度
        output, hidden = self.gru(embeded)  # hidden:[num_layer*num_directions, batch_size, hidden_size] [1, 128, 64]
        output, output_length = pad_packed_sequence(output, batch_first=True, padding_value=config.num_sequence.PAD)  # output:[batch_size, max_len, num_directions*hidden_size] [128, 9, 1*64]

        return output, hidden


if __name__ == '__main__':
    from data.dataset import train_dataloader
    numencoder = NumEncoder()
    print(numencoder)
    for input, label, input_length, label_length in train_dataloader:
        print(input_length)
        output, hidden = numencoder(input, input_length)
        print(output, output.size())
        print(hidden, hidden.size())
        break