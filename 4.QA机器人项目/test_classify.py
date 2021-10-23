# -*- coding: utf-8 -*-
# date: 2021/10/21
# Project: Pytorch学习
# File Name: test_classify.py
# Description: 测试分类模型
# Author: Anefuer_kpl
# Email: 374774222@qq.com

from classify.classify_model import Classify
from utils.cut_sentence import cut

if __name__ == '__main__':
    classify = Classify()
    classify.test()