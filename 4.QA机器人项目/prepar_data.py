# -*- coding: utf-8 -*-
# date: 2021/10/21
# Project: Pytorch学习
# File Name: prepar_data.py
# Description: 作为生成数据的主函数使用，通过调用已经写好的方法直接生成所需要的数据
# Author: Anefuer_kpl
# Email: 374774222@qq.com

from data.prepar_corpus.prepar_classify_corpus import process_classify_xiaohuangji, process_classify_QA
from data.prepar_corpus.prepar_QA_data import create_QA_data
from data.prepar_corpus.prepar_chat_data import prepar_chat_xiaohuangji
from utils.cut_sentence import cut



if __name__ == '__main__':
    # print(cut('是王若猫的。'))
    create_QA_data()

    process_classify_xiaohuangji()
    process_classify_QA()

    prepar_chat_xiaohuangji()