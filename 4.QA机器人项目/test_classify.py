# -*- coding: utf-8 -*-
# date: 2021/10/21
# Project: Pytorch学习
# File Name: test_classify.py
# Description: 测试分类模型
# Author: Anefuer_kpl
# Email: 374774222@qq.com

from classify.classify_model import build_classify_model, get_classify_model
from utils.cut_sentence import cut

if __name__ == '__main__':
    build_classify_model()

    # 测试模型
    text = [
        '我肚子非常疼',  # chat
        '有什么办法治疗我的病么', # QA
        '这太糟糕了',  # chat
        '我心情不太好',   # chat
        '你在开玩笑吧',   # chat
        '为什么会得这个病', # QA
        '告诉我该怎么准备好不好',  # QA
        '天哪，求求你快告诉我吧',  # QA
    ]

    model = get_classify_model()
    text_cut = []
    for content in text:
        text_cut.append(' '.join(cut(content)))

    # print(text_cut)

    pred = model.predict(text_cut)
    print('预测结果',pred)