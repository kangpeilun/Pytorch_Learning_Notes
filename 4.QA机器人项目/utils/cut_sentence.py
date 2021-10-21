#-*- coding:utf-8 -*-
# author: Anefuer_kpl
# Email: 374774222@qq.com
# datatime: 2021-10-18 10:04
# project: Pytorch学习

'''分词'''
import logging

import jieba
import jieba.posseg as psg  # 用于返回词性
import config
import string

# 关闭jieba log输出
jieba.setLogLevel(logging.INFO)
# 加载用户词典
jieba.load_userdict(config.user_dict_path)
# 准备英文字符
# 需要识别到一起的特殊字符
letters = "+"
letters = string.ascii_lowercase+letters

def cut_sentence_by_word(sentence):
    '''
    实现中英文分词
    python和c++哪个个难？  --> [python, 和, c++, 哪, 个, 难, ？]
    '''
    result = []
    temp = ""
    for word in sentence:  # 对句子中一个字符一个字符的判断
        word = word.lower()
        if word in letters:  # 如果当前字符是letters(字母)则单独保存起来
            temp += word    # 将连续的多个字母连接起来拼成一个词语
        else:
            if temp != "":  # 如果temp不为空，表示有英文单词，此时把连起来的英文单词保存到
                result.append(temp)
                temp = ""  # 将temp置为空，重新添加新的词语
            result.append(word)  # 同时将非英文单词添加进去

    if temp != "":  # 当英文字符出现在最后的位置，防止被遗漏
        result.append(temp)

    return result


def cut(sentence, by_word=False, use_stopwords=False, with_sg=False):
    '''
    :param sentence: 句子
    :param by_word: 是否按照单个字分词, 即每一个字都当作一个词(一个完整的单词也作为一个字)
    :param use_stopwords: 是否使用停用词
    :param with_sg: 是否返回词性
    :return:
    '''
    if by_word==True:  # 是否按照单个字分词
        result = cut_sentence_by_word(sentence)
    else:
        result = psg.lcut(sentence)  # 用于返回词性
        result = [(i.word, i.flag) for i in result]  # 将pair对象转换为我们熟悉的格式，得到带有词性的元组
        if not with_sg:  # 不返回词性
            result = [i[0] for i in result]  # 不返回词性的话，只需要取第一个索引值即可

    if use_stopwords: # 使用停用词
        if not with_sg: # 不返回词性，则需要单独处理
            result = [i[0] for i in result if i[0] not in config.stopwords]  # 只保留不在停用词中的词语
        else:  # 返回词性
            result = [i for i in result if i[0] not in config.stopwords]

    return result



if __name__ == '__main__':
    # print('p' in letters)
    cut_sentence_by_word('python和c++哪个难？')