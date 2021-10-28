# -*- coding: utf-8 -*-
# date: 2021/10/25
# Project: Pytorch学习
# File Name: decoder_model.py
# Description: 
# Author: Anefuer_kpl
# Email: 374774222@qq.com
import torch
from torch import nn
import torch.nn.functional as F
import config
import random

class NumDecoder(nn.Module):
    def __init__(self):
        super(NumDecoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=config.num_embedding,
                                      embedding_dim=config.embedding_dim,
                                      padding_idx=config.num_sequence.PAD)
        self.gru = nn.GRU(input_size=config.embedding_dim,
                          hidden_size=config.hidden_size,
                          num_layers=config.num_layer,
                          batch_first=True)
        self.fc = nn.Linear(config.hidden_size, config.num_embedding)  # 输入hidden大小，输出 所有词的种类数


    def forward(self, label, encoder_hidden):
        '''
        @param label: [batch_size, max_len+1]
        @param encoder_hidden: [1, batch_size, hidden_size]
        @return:
        '''
        # 1.获取encoder的输出，作为decoder第一次的hidden_state
        decoder_hidden = encoder_hidden
        # 2.准备decoder第一个时间步的输入: [batch_size, 1] SOS作为输入
        '''
            因为要告诉解码器开始解码，且我们定义只要遇到SOS就表示句子的开始，且每一个时间步(词)都要单独计算一次，不同于编码器在最后一步才生成结果
            因此形状要和batch_size相同，故为[batch_size,1]，即batch_size个数据
            torch.LongTensor([[SOS]]*batch_size)
        '''
        decoder_input = torch.LongTensor([[config.num_sequence.SOS]]*config.train_batch_size).to(config.device)
        # 3.在第一个时间步上进行计算，得到第一个时间步的输出 和 hidden_state
        # 4.把前一个时间步的输出进行计算，得到第一个最后的输出结果
        # 5.把前一次的hidden_state 作为当前时间步的hidden_state的输入，把前一次的输出，作为当前时间步的输入
        # 6.循环4-5步骤

        # 保存预测的结果
        '''
            因为是对 label进行操作，因为我们的input和label的要求是: label就是在input后面加个0，所以label的长度比input要多1，故max_len+1
            input: 123
            lable: 1230
            PS: [config.train_batch_size, config.max_len+1, config.num_embedding] <==> [batch_size, max_len+1, vocab_size]
        '''
        decoder_outputs = torch.zeros([config.train_batch_size, config.max_len+1, config.num_embedding]).to(config.device)

        for t in range(config.max_len+1):  # 这里的max_len+1同上理
            # decoder_output_t: [batch_size, vocab_size]   decoder_hidden: [1, batch_size, hidden_size]
            # decoder_output中保存的就是每个batch中每个句子中 每个词语的 预测概率
            decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            # 保存decoder_output_t到decoder_outputs
            '''
                decoder_output_t 可以理解为每一个时间步上的输出，即 一个一个的词语，而不是整个句子
                而我们要得到的是 对于整个句子的 预测结果，因此需要 每一个词语的预测结果进行合并，最终得到整个句子的预测结果
                example:
                    decoder_output_t 预测的结果为：
                        batch_size 可以理解为 句子的个数
                        max_len表示句子的长度，也就是每个句子中 词语的个数
                        句子1 的 第1个 词的预测结果：[batch_size_1, vocab_size_1]
                        句子1 的 第2个 词的预测结果：[batch_size_1, vocab_size_2]
                        句子1 的 第3个 词的预测结果：[batch_size_1, vocab_size_3]
                                        ......
                        句子1 的 第3个 词的预测结果：[batch_size_1, vocab_size_(max_len+1)]
                        句子2 的 第1个 词的预测结果：[batch_size_2, vocab_size_1]
                    因为是并行计算的，decoder_output_t可以理解为一次性算出来 batch_size 个 vocab_size_1，以此类推
                
                理解: decoder_outputs[:, t, :] = decoder_output_t 
                    decoder_outputs的形状为[batch_size, max_len+1, vocab_size]
                        可以理解为：有batch_size条句子，每个句子的长度为max_len+1，每个句子中 每个词语 的预测值有 vocab_size个预测结果
                    decoder_output_t形状为[batch_size, vocab_size]
                        可以理解为：有batch_size条句子，每个句子中 每个词语 的预测值有 vocab_size个预测结果
            '''
            decoder_outputs[:, t, :] = decoder_output_t
            if random.random() > config.teacher_forcing_ratio:
                '''
                    当随机数大于阈值时，使用teacher_forcing机制
                        将真实值传给传给下一个time_step，这样能够加速模型的收敛
                        
                    decoder_input = label[t]
                        因为decoder是一个time_step一个time_step的进行预测的，因此使用teacher_forcing机制传给下一个time_step的
                        应该是一个 真实的词，而不是一个 完整的句子
                        
                    label[:, t]表示取每一行的第t列的元素，但得到的是一维的，而我们想要的是 二维的 [batch_size, 1]
                    因此使用unsqueeze(-1)，人为的在最后添加一个维度
                '''
                decoder_input = label[:, t].unsqueeze(-1)
                # print('teacher_forcing',decoder_input, decoder_input.size())
            else:
                '''
                    torch.topk(decoder_output, 1) 获取tensor中最大的一个值, 默认从tensor的最后一维获取数据
                    input：就是输入的tensor，也就是要取topk的张量
                    k：就是取前k个最大的值。
                    dim：就是在哪一维来取这k个值。
                    lagest：默认是true表示取前k大的值，false则表示取前k小的值
                    sorted：是否按照顺序输出，默认是true。
                    out ： 可选输出张量 (Tensor, LongTensor)
                    返回值:
                        value: 最大值的数值
                        index: 最大值所在索引位置
                '''
                value, index = torch.topk(decoder_output_t, 1)
                '''
                    因为forward_step的输出是每种词的概率，所以最大值的索引位置可以理解为 num_sequence中构造的inverse_dict对应的key
                    通过 索引 就可以映射出原来的 词 是什么
                    
                    小于阈值时，不使用teacher_forcing机制，将当前time_step的预测值传给像一个time_step
                '''
                decoder_input = index
                # print('no teacher_forcing',decoder_input, decoder_input.size())

        return decoder_outputs, decoder_hidden

    def forward_step(self, decoder_input, decoder_hidden):
        '''
        计算每一个时间步（词）上的结果
        @param decoder_input: [batch_size, 1]
        @param decoder_hidden: 上一步的隐藏层输出 [1, batch_size, hidden_size]
        @return:
        '''
        decoder_input_embeded = self.embedding(decoder_input)  # [batch_size, 1, embedding_dim] 因为每一个词单独预测一次，故句子的长度seq_len为 1
        output, decoder_hidden = self.gru(decoder_input_embeded, decoder_hidden)
        # output: [batch_size, 1, hidden_size]
        # decoder_hidden: [1, batch_size, hidden_size]
        output = output.squeeze(1) # 将维度值为1 的维度 去除  [batch_size, hidden_size]
        output = self.fc(output)  # [batch_size, vocab_size]  vocab_size表示所有词的种类数
        output = F.log_softmax(output, dim=-1)  # [batch_size, vocab_size] 得到当前时间步的输出，softmax之后输出的是概率
        # print('output size:',output.size())
        return output, decoder_hidden

    def evaluate(self, encoder_hidden):
        '''
        模型评估时，decoder使用该方法
        @param encoder_hidden: [1, batch_Size, hidden_size]
        @return:
        '''
        decoder_hidden = encoder_hidden
        batch_size = encoder_hidden.size(1)
        decoder_input = torch.LongTensor([[config.num_sequence.SOS]]*config.test_batch_size).to(config.device)

        indices = []  # 存放整个句子中每个词语的预测结果, 也就是模型的预测结果
        '''
            因为现在是预测过程，所以并不能通过句子的长度来判断 是否预测完毕
            而应该通过 预测的 词语是否为结束符来 判断是否终止循环
        '''
        for i in range(config.max_len+5):
            # decoder_output_t: [batch_size, vocab_size]
            decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            value, index = torch.topk(decoder_output_t, 1)  # index: [batch_size, 1]
            decoder_input = index
            '''
                模型一次性会预测出batch_size个结果，每一次的index都是一个时间步上的预测结果
                而 indices 则是将所有时间步上的预测结果合并起来
                indices的形状：
                    [
                        timestep_1: [batch_size_1, batch_size_2, batch_size_3, ......],
                        timestep_2: [batch_size_1, batch_size_2, batch_size_3, ......],
                        timestep_3: [batch_size_1, batch_size_2, batch_size_3, ......],
                        timestep_4: [batch_size_1, batch_size_2, batch_size_3, ......],
                            ......
                    ]
            '''

            index = index.squeeze(1)  # squeeze(1) 将第1个维度的1去掉，这样 每一行 就相当于是一个time_step
            indices.append(index.numpy())

        return indices