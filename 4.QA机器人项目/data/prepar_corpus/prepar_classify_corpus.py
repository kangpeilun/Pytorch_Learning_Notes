# -*- coding:utf-8 -*-
# author: Anefuer_kpl
# Email: 374774222@qq.com
# datatime: 2021-10-18 14:26
# project: Pytorch学习

'''准备分类语料'''

# 使用多线程处理数据
from concurrent.futures import ThreadPoolExecutor

import config
import pandas as pd
from utils.cut_sentence import cut
from tqdm import tqdm
import random

'''
    每次从该list中随机取一个，如果取出的值为1，则划为测试集
    这样就可以巧妙的控制划分的比例
    train/test = 3/1
    即有25%的数据作为测试集
'''
train_test_split = [0,0,0,1]

def keywords_in_line(line):
    # 判断line中是否存在不符合要求的词
    keyword_list = [
        '中药', '西药', '医院', '医生', '诊所', '手术失败', '死亡', '残疾', '残废', '瘫痪',
        '植物人', '大出血', '无效', '传染', '感染', '羊水栓塞', '脑瘫', '畸形', '看病', '吃药',
        '感冒', '头疼', '肚子疼', '拉肚子', '咳嗽', '肚子疼', '西药', '热水', '休息', '放松',
    ]
    for word in line:  # 对于当前句子判断是否含有以上关键词
        if word in keyword_list:
            return True
        else:
            return False


def process_one_line_xiaohuangji(line, train_file, test_file):
    '''
    处理小黄鸡闲聊语料
    :param line: 每一行句子
    :param file: 要写入的文件
    :return: M 是王若猫的。--> 是 王 若 猫 的 。\t__label__chat
    '''
    # TODO 句子长度为1时，考虑删除
    if line.startswith('E'):
        return
    elif line.startswith('M'):
        line = line.split(' ')[-1].replace('\n', '')  # 根据空格切分，取出最后的内容
        # if keywords_in_line(line_cut):  # 判断该句子是否含有关键字
        line_cut = ' '.join(cut(line)) + '\t' + '__label__chat'  # 更改数据格式
        if random.choice(train_test_split) == 1:  # 如果随机选择的数据为1，则放入测试集；否则放入训练集
            test_file.write(line_cut + '\n')
        else:
            train_file.write(line_cut + '\n')


def process_xiaohuangji():
    '''
    处理小黄鸡闲聊语料
    :return: M 是王若猫的。--> 是 王 若 猫 的 。\t__label__chat
    '''
    classify_train_file = open(config.classify_train_path, 'a+', encoding='utf-8')
    classify_test_file = open(config.classify_test_path, 'a+', encoding='utf-8')
    with open(config.xiaohuangji_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        with ThreadPoolExecutor(1000) as t:  # 起100个线程
            for line in tqdm(lines, total=len(lines), ascii=True):
                t.submit(process_one_line_xiaohuangji, line=line, train_file=classify_train_file, test_file=classify_test_file)  # 把写入任务提交给线程池
    classify_train_file.close()
    classify_test_file.close()
    print('闲聊语料处理完毕！')


def process_one_line_QA(line, train_file, test_file):
    '''
    处理QA问答语料
    :return:
    '''
    content_cut = ''.join(cut(line.strip().replace('\n', '')))  #
    # if keywords_in_line(content_cut):
    content_cut = ' '.join(cut(content_cut)) + '\t' + '__label__QA'
    if random.choice(train_test_split) == 1:    # 如果随机选择的数据为1，则放入测试集；否则放入训练集
        train_file.write(content_cut + '\n')
    else:
        test_file.write(content_cut + '\n')

def process_QA():
    classify_train_file = open(config.classify_train_path, 'a+', encoding='utf-8')
    classify_test_file = open(config.classify_test_path, 'a+', encoding='utf-8')
    questions = pd.read_csv(config.question_path)
    contents = questions['content'].tolist()  # 获取content列所有的数据，并将结果转换为list
    with ThreadPoolExecutor(1000) as t:  # 起100个线程
        for content in tqdm(contents, total=len(contents), ascii=True):
            t.submit(process_one_line_QA, line=content, train_file=classify_train_file, test_file=classify_test_file)
    classify_train_file.close()
    classify_test_file.close()
    print('QA语料处理完毕！')


if __name__ == '__main__':
    pass
