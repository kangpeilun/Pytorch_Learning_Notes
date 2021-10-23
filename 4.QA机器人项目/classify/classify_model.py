# -*- coding: utf-8 -*-
# date: 2021/10/21
# Project: Pytorch学习
# File Name: classify_model.py
# Description: 构建分类模型
# Author: Anefuer_kpl
# Email: 374774222@qq.com

from utils.cut_sentence import cut
import fasttext
import config


class Classify():
    def __init__(self):
        self.train_data = config.classify_train_path
        self.test_data = config.classify_test_path

    def train(self):
        '''
        train_data 为分词后的数据: 你 还 好 么 ？\t__label__chat
        :return:
        '''
        model = fasttext.train_supervised(self.train_data, epoch=20, wordNgrams=1, minCount=5)  # 加载训练数据进行训练
        model.save_model(config.classify_model_path)

    def test(self):
        model = fasttext.load_model(config.classify_model_path)
        test_data =[]
        label_data = []
        for line in open(self.test_data, encoding='utf-8').readlines():
            text = line.split('\t')[0]
            label = line.split('\t')[1]
            test_data.append(text)
            label_data.append(label)

        lables, scores = model.predict(test_data)  # 加载测试数据进行预测
        sum = 0
        for label,pre_label in zip(label_data, lables):
            if label.replace('\n', '') == pre_label[0]:
                sum+=1
        print('acc:', round(sum/len(test_data),4))

    def __is_QA(self, label, score):
        '''
            预测结果的判定由人为来决定
            __label__chat 和 __label__QA 的分数都有可能很高，但为了通过分数区分，我们人为的缩小__label__chat的分数值
            相当于，我们把得分小的作为 __label__chat，而得分高的作为 __label__QA
        '''
        if label[0] == '__label__chat':
            score = 1 - score

        if score[0] >= 0.95:
            return '__label__QA'
        else:
            return '__label__chat'

    def predict(self, sentence):
        model = fasttext.load_model(config.classify_model_path)
        sentence = ' '.join(cut(sentence))
        label, score = model.predict(sentence)
        pred = self.__is_QA(label, score)
        print('预测结果:', pred)
        return pred
