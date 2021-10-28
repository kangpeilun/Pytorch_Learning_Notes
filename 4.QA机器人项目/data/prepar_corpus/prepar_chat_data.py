# -*- coding: utf-8 -*-
# date: 2021/10/28
# Project: Pytorch学习
# File Name: prepar_chat_data.py
# Description: 准备闲聊语料
# Author: Anefuer_kpl
# Email: 374774222@qq.com

from tqdm import tqdm
import config
import random
import string

from utils.cut_sentence import cut

train_test_split = [0,0,0,1]

def filter(line):
    '''
    用于过滤不符合要求的句子
    @param line:
    @return:
    '''
    if line.lower() in list(string.ascii_lowercase):
        # 句子为单个字母，过滤掉
        return True
    if len(line)<2:
        # 句子长度小于2，过滤掉
        return True


def prepar_chat_xiaohuangji(by_word=False):
    '''
    处理小黄鸡闲聊语料
    @return:
    '''
    train_file = open(config.chat_train_path, 'a+', encoding='utf-8')
    test_file = open(config.chat_test_path, 'a+', encoding='utf-8')

    '''
        is_M使用一个标识符，来判断是否为M
        当遇到E是，is_M=0
        遇到第一个M，is_M=1
        遇到第二个M，is_M=2
    '''
    is_M = 0
    train_or_test = 0  # 用于标记该条数据用于训练还是测试，默认为0 表示用于训练集，1表示用于测试
    with open(config.xiaohuangji_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, total=len(lines), ascii=True, desc='处理小黄鸡闲聊语料'):
            if line.startswith('E'):
                is_M = 0    # 当遇到 E 时，M就进行重置
                train_or_test = random.choice(train_test_split)   # 当遇到 E 的时候就重新选择一下
                continue

            is_M += 1
            line = line.replace('\n', '').strip()[1:]  # 去掉每句话后面的回车
            if filter(line):
                # 满足过滤条件的，就跳过该句子
                continue

            line = cut(line) if by_word else cut(line, by_word=by_word)
            line = " ".join(line)
            if train_or_test == 0:
                # 本次的样本用于训练
                if is_M == 1:
                    # 该条语句是问题
                    train_file.write(line+'\t')
                elif is_M == 2:
                    # 该条语句是回答
                    train_file.write(line+'\n')

            elif train_or_test == 1:
                # 本次样本用于测试
                if is_M == 1:
                    # 该条语句是问题
                    test_file.write(line + '\t')
                elif is_M == 2:
                    # 该条语句是回答
                    test_file.write(line + '\n')

    train_file.close()
    test_file.close()

if __name__ == '__main__':
    pass