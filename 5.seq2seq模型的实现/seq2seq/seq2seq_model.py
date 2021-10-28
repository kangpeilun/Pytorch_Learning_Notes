# -*- coding: utf-8 -*-
# date: 2021/10/27
# Project: Pytorch学习
# File Name: seq2seq_model.py
# Description: 把encoder和decoder进行合并，得到seq2seq模型
# Author: Anefuer_kpl
# Email: 374774222@qq.com

from torch import nn
from encoder.encoder_model import NumEncoder
from decoder.decoder_model import NumDecoder

class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = NumEncoder()
        self.decoder = NumDecoder()

    def forward(self, input, label, input_length, label_length):
        '''
        @param input: [batch_size, max_len]
        @param input_length: 每个batch中句子的长度
        @return:
        '''
        # encoder_outputs: [batch_size, max_len, num_directions*hidden_size]
        # encoder_hidden: [num_layer*num_directions, batch_size, hidden_size]
        encoder_outputs, encoder_hidden = self.encoder(input, input_length)

        # decoder_outputs: [batch_size, max_len, vacab_size]
        # decoder_hidden: [num_layer*num_directions, batch_size, hidden_size]
        decoder_outputs, decoder_hidden = self.decoder(label, encoder_hidden)

        return decoder_outputs, decoder_hidden

    def evaluate(self, input, input_length):
        encoder_outputs, encoder_hidden = self.encoder(input, input_length)
        indices = self.decoder.evaluate(encoder_hidden)

        return indices