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
        test_data = [text.split('\t')[0] for text in open(self.test_data, encoding='utf-8').readlines()]
        lables, scores = model.predict(test_data)  # 加载测试数据进行预测
        # lables [['__label__chat'], ['__label__chat'], ['__label__chat'], ['__label__chat'], ['__label__chat'], ['__label__chat'], ['__label__chat'], ['__label__chat'], ['__label__chat'], ['__label__chat']]
        # scores [array([1.00001], dtype=float32), array([1.0000099], dtype=float32), array([1.00001], dtype=float32), array([1.00001], dtype=float32), array([1.00001], dtype=float32), array([1.00001], dtype=float32), array([1.00001], dtype=float32), array([1.00001], dtype=float32), array([0.99937207], dtype=float32), array([1.00001], dtype=float32)]
        print('acc:', (sum(scores) / len(test_data))[0])

    def is_QA(self, label, score):
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
        pred = self.is_QA(label, score)
        print('预测结果:', pred)
        return pred


def build_classify_model():
    '''
    wordNgrams=1 每连续的1个字作为一个词语
    minCount=5
    :return:
    '''
    model = fasttext.train_supervised(config.classify_train_path, epoch=20, wordNgrams=1, minCount=5)
    model.save_model(config.classify_model_path)


def get_classify_model():
    '''加载模型'''
    model = fasttext.load_model(config.classify_model_path)
    return model


def eval():
    pass
