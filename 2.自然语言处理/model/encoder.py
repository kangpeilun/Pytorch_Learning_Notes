# -*- coding: utf-8 -*-
# date: 2021/10/23
# Project: Pytorch学习
# File Name: encoder.py
# Description: 编码器
# Author: Anefuer_kpl
# Email: 374774222@qq.com

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence  # 加速GRU或LSTM的训练过程
from torch import nn
import config


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        '''
            指定padding_idx=0，因为PAD为填充符号，不需要对其进行更新, 加速训练过程
        '''
        self.embedding = nn.Embedding(num_embeddings=len(config.num_sequence),
                                      embedding_dim=config.embedding_dim,
                                      padding_idx=config.num_sequence.PAD,)
        self.gru = nn.GRU(input_size=config.embedding_dim,
                          num_layers=config.num_layer,
                          hidden_size=config.hidden_size,
                          batch_first=True)

    def forward(self, input, input_length):
        '''

        @param input: [batch_size, max_len]
        @return:
        '''
        embeded = self.embedding(input) # [batch_size, max_len, embedding_dim] [128, 9, 100]
        embeded = pack_padded_sequence(embeded, input_length, batch_first=True)  # 把embedding后的结果打包
        output, hidden = self.gru(embeded)
        output, output_length = pad_packed_sequence(output, batch_first=True, padding_value=config.num_sequence.PAD, total_length=config.max_len)    # 把gru的输出解包 output:[batch_size, max_len, num_directions*hidden_size] [128, 9, 1*64]

        return output, hidden, output_length


if __name__ == '__main__':
    from data.dataset import train_dataloader
    encoder = Encoder()
    print(encoder)
    for input, target, input_length, label_length in train_dataloader:
        output, hidden, output_length = encoder(input, input_length)
        print(output.size())
        print(hidden.size())
        print(output_length)
        break


