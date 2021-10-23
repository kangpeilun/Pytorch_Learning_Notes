# -*- coding: utf-8 -*-
# date: 2021/10/23
# Project: Pytorch学习
# File Name: num_sequence.py
# Description: 将字符 和 数字进行映射
# Author: Anefuer_kpl
# Email: 374774222@qq.com


class Num_sequence():
    '''
    自然语言处理常用标识符:
        <UNK>: 低频词或未在词表中的词
        <PAD>: 补全字符
        <GO>/<SOS>: 句子起始标识符
        <EOS>: 句子结束标识符
        [SEP]：两个句子之间的分隔符
        [MASK]：填充被掩盖掉的字符
    '''
    PAD_TAG = 'PAG'
    UNK_TAG = 'UNK'
    SOS_TAG = 'SOS'  # 句子开始符号
    EOS_TAG = 'EOS'  # 结束符

    PAD = 0
    UNK = 1
    SOS = 2
    EOS = 3

    def __init__(self):
        self.dict = {
            self.PAD_TAG: self.PAD,
            self.UNK_TAG: self.UNK,
            self.SOS_TAG: self.SOS,
            self.EOS_TAG: self.EOS,
        }
        for i in range(10):
            self.dict[str(i)] = len(self.dict)

        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))  # 快速得到反转字典

    def transform(self, sentence, max_len, add_eos=False):
        '''
           把sentence转化为数字序列
           sentence: ['1', '2', '4', ...] --> [1, 2, 4, ...]
        '''
        # 11, 10
        if len(sentence) >= max_len:  # 句子大于最大长度，需要进行裁剪
            '''
                这里减1的目的是使得句子的长度保持一致
                以 句子长度=11， 最大长度=10 为例:
                    因为 EOS也相当一个字符，使用代码sentence = sentence[:max_len]对句子进行切割后，再加上EOS
                    那么 句子的长度就会超过 最大限制10
                    
                    因此 使用 max_len-1 
                    做法就是提前留出一个位置给 EOS
                    这样就能保证句子的最大长度总是10 
            '''
            sentence = sentence[:max_len - 1]
            if add_eos:
                sentence = sentence + [self.EOS_TAG]
        # 9, 10
        else:
            if add_eos:
                sentence = sentence + [self.EOS_TAG]
            # 如果句子小于最大长度， 则需要在句子后面补充PAD占位符
            sentence = sentence + [self.PAD_TAG] * (max_len - len(sentence))

        # 如果词语未在字典中出现则用UNK替代
        result = [self.dict.get(i, self.UNK) for i in sentence]
        return result

    def inverse_transform(self, indices):
        '''把序列转回字符串'''
        return [self.inverse_dict.get(i, self.UNK_TAG) for i in indices]

    def __len__(self):
        return len(self.dict)


if __name__ == '__main__':
    numseq = Num_sequence()
    print(numseq.dict)
    print(numseq.inverse_dict)
