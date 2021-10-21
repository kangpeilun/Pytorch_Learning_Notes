# -*- coding:utf-8 -*-
# author: Anefuer_kpl
# Email: 374774222@qq.com
# datatime: 2021-09-25 10:29
# project: Pytorch学习

'''
    构建词典，实现方法把句子转化为数字序列和其翻转
'''


class Word2Seq():
    UNK_TAG = 'UNK'  # UNK表示特殊字符，没见见过的词语都用UNK代替，UNK对应数字0
    PAD_TAG = 'PAD'  # 把短句子进行填充，使用PAD进行填充，PAD对应数字为1

    UNK = 0
    PAD = 1

    def __init__(self):
        # 将 词语 和 编号对应起来
        self.dict = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }
        self.count = {}  # 统计词频

    def fit(self, text):
        '''
        把单个句子保存到dict中, 并统计每个词语的词频
        :param text: [word1, word2, word3, ...]
        :return:
        '''
        for word in text:
            '''Tips： 编程技巧
                self.count.get(word, 0) + 1
                如果当前字典中'word'存在则返回key对应的值并+1，如果'word'不存在则返回0+1
            '''
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self, min=5, max=None, max_features=None):
        '''
        生成词典, 剔除不符合数量要求的词语
        :param min: 词语最小出现次数
        :param max: 最大出现次数
        :param max_features: 一共保留多少个词语
        :return:
        '''
        # 删除count中词频小于min的word
        if min is not None:
            '''
                PS: 遍历字典时，其实遍历的是key
                self.count.items() 返回的是元组 (key, value)
            '''
            self.count = {word: value for word, value in self.count.items() if value >= min}
        # 删除count中词频大于max的word
        if max is not None:
            self.count = {word: value for word, value in self.count.items() if value <= max}
        # 限制保留的词语数
        if max_features is not None:
            '''
                sorted后会将元组变成列表
                self.count.items() 是一个可迭代对象, 其中的每一个值是一个(key,value)对
                key=lambda x:x[-1] 使字典中的key根据items中的value进行排序, x[-1]表示取最后一个值也就是value
                reverse=True 由大到小，降序排列
                [:max_features] 将排序后的前 max_features 个数取出来(因为sorted已经将dict_items变为list，故可以这样取值)
            '''
            temp = sorted(self.count.items(), key=lambda x: x[-1], reverse=True)[
                   :max_features]  # 这样得到的是一个列表，其中每个元素是一个二值元组
            self.count = dict(temp)  # 将[(key, value), (key, value)] 转化为 {key:value, key:value}

        # 给每一个词语进行编号  {word:num}
        for word in self.count:
            '''
                因为原来的self.dict中已有self.UNK_TAG: self.UNK 和 self.PAD_TAG: self.PAD 两组键值对
                故新词的编号从 2 开始，也就不会和之前的重复
            '''
            self.dict[word] = len(self.dict)

        # 得到一个翻转的dict词典 {num:word}
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, text, max_len=None):
        '''
        把句子转换为序列
        :param text: [word1, word2, ...]
        :param max_len: int, 句子词语最大的个数，并对句子进行填充或裁剪
        :return: [1, 2, 4, ...]
        '''
        '''
            在self.dict中找到句子中每一个词语对应的编号，组成list返回
        '''
        if max_len is not None:
            if max_len > len(text):
                # 如果句子长度小于max_len则对句子填充max_len-len(text)个'PAD'
                text = text + [self.PAD_TAG] * (max_len - len(text))
            else:
                # 如果句子长度大于max_len, 则对句子裁剪，取前max_len个
                text = text[:max_len]

        return [self.dict.get(word, self.UNK) for word in text]

    def inverse_transform(self, indices):
        '''
        将序列转化为句子
        :param indices: [1, 2, 4, 5, 3, ...]
        :return: [word1, word2, word4, word3, ...]
        '''
        return [self.inverse_dict.get(index) for index in indices]

    def __len__(self):
        # 不重复的词语的个数
        return len(self.dict)


if __name__ == '__main__':
    # ws = Word2Seq()
    # ws.fit(['我','是','谁'])
    # ws.fit(['我','是','康佩伦'])
    # ws.build_vocab(min=1)  # 每个词语至少出现一次
    # print(ws.dict)
    #
    # ret = ws.transform(['我','爱','kpl'])
    # print(ret)
    # ret = ws.inverse_transform(ret)
    # print(ret)

    # fit_save_word_seq(10000)
    pass
