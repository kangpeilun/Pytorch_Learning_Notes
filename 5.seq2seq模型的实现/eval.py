# -*- coding: utf-8 -*-
# date: 2021/10/27
# Project: Pytorch学习
# File Name: eval.py
# Description: 模型评估
# Author: Anefuer_kpl
# Email: 374774222@qq.com

import torch
from seq2seq.seq2seq_model import Seq2Seq
from data.dataset import test_dataloader
import config
from tqdm import tqdm
import numpy as np

# 测试流程
# 1.准备测试数据
# 2.加载模型
# 3.获取预测值
# 4.反序列化，观察结果

seq2seq = Seq2Seq().to(config.device) # 实例化模型
seq2seq.load_state_dict(torch.load(config.model_save_path))  # 加载模型

def eval():
    bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), ascii=True)
    bar.set_description('eval')
    for index, (input, label, input_length, label_length) in bar:
        input = input.to(config.device)
        input_length = input_length.to(config.device)
        indices = seq2seq.evaluate(input, input_length)
        indeces = np.array(indices).transpose()  # 转置，因为seq2seq.evaluate的返回值中 每一列 表示一个完整的句子

        results = []  # 存放所有的预测结果unsqueeze(-1)
        labels = []  # 存放所有的标签
        # 将预测结果 和 label 反序列化
        for line_indeces, line_label in zip(indeces, label):
            '''
                因为在dataloader中我们把label处理成tensor类型了，所以这里需要将tensor再转换为numpy数组，这样才能正常根据索引取出其中的值
            '''
            line_label = line_label.numpy()
            temp_result_indeces = config.num_sequence.inverse_transform(line_indeces)
            temp_result_label = config.num_sequence.inverse_transform(line_label)

            cur_line_indeces = ""
            cur_line_label = ""
            for word in temp_result_indeces:
                if word in [config.num_sequence.EOS_TAG, config.num_sequence.PAD_TAG]:
                    break
                cur_line_indeces += word
            results.append(''.join(cur_line_indeces))

            for word in temp_result_label:
                if word == config.num_sequence.EOS_TAG:
                    break
                cur_line_label += word
            labels.append(cur_line_label)

    '''
        使用numpy 将预测值 和 真实值 进行比较，并返回两个矩阵中对应元素是否相等的布尔值
        布尔值可以直接求和
    '''
    total_correct = sum(np.array(results)==np.array(labels))  # 得到预测正确的个数
    acc = total_correct/len(results)
    print('模型的预测准确率为:{:.3f}'.format(acc))


if __name__ == '__main__':
    eval()