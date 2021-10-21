#-*- coding:utf-8 -*-
# author: Anefuer_kpl
# Email: 374774222@qq.com
# datatime: 2021-10-16 11:45
# project: Pytorch学习

'''
    配置文件
'''

# 准备语料的配置文件
user_dict_path = 'data/corpus/user_dict/user_dict.txt'      # 用户词典
stopwords_path = 'data/corpus/user_dict/stopwords/baidu_stopwords.txt'  # 停用词
stopwords = set([i.strip() for i in open(stopwords_path, encoding='utf-8').readlines()])  # 停用词列表

# 问答语料
answer_path = 'data/corpus/QA_data/answer.csv'  # 答案
question_path = 'data/corpus/QA_data/question.csv' # 问题

# 闲聊语料
xiaohuangji_path = 'data/corpus/chat_data/xiaohuangji_weifenci.conv'  # 小黄鸡未分词

# classify fasttext分类语料准备——闲聊语料 以及 QA语料
classify_train_path = 'data/corpus/classify/classify_train.txt'  # 训练数据
classify_test_path = 'data/corpus/classify/classify_test.txt'   # 测试数据

#=====================================分类相关===========================================
classify_model_path = 'model/classify.pt'   # 分类模型保存的路径