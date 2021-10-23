# -*- coding:utf-8 -*-
# author: Anefuer_kpl
# Email: 374774222@qq.com
# datatime: 2021-09-24 13:07
# project: Pytorch学习

import torch
from torch.utils.data import Dataset, DataLoader
import os
from utils.dataset import tokenlize
from utils.lib import ws

'''
    构建自己的数据集
'''

MAX_LEN = 20

# 数据集对象
class ImdbDataset(Dataset):
    def __init__(self, train=True):
        # 通过train来判断加载哪一个数据集
        super(ImdbDataset, self).__init__()
        self.train_data_path = r'./data/aclImdb/train'
        self.test_data_path = r'./data/aclImdb/test'
        data_path = self.train_data_path if train else self.test_data_path

        # 把所有的文件名称放入列表
        temp_data_path = [os.path.join(data_path, 'pos'), os.path.join(data_path, 'neg')]
        self.total_file_path = []  # 存放所有的数据文件, 积极的 和 消极的
        for path in temp_data_path:
            file_name_list = os.listdir(path)  # 获取该路径下所有的文件名
            file_path_list = [os.path.join(path, file_name) for file_name in file_name_list if
                              file_name.endswith('.txt')]  # 使用生成表达式，获取该文件夹下所有数据文件的完整路径。过滤掉不是txt的数据
            '''
                注意：这里是extend, 将其他列表中的值添加到当前列表中
            '''
            self.total_file_path.extend(file_path_list)

    def __getitem__(self, item):
        file_path = self.total_file_path[item]
        label_num = file_path.replace('.txt', '').rsplit('_', 1)[-1]
        label = 1 if int(label_num) > 4 else 0  # 当label>4表示积极，用数值0表示积极
        text = tokenlize(open(file_path, encoding='utf-8').read())
        return text, label

    def __len__(self):
        return len(self.total_file_path)


#***PS：当Dataset中返回的input结果时字符串的时候，可以通过修改collate_fn解决***
# collate_fn的作用是对每一个batch进行处理
# 因为torch默认的default_collate在处理文本时会出错，故这里对collate_fn方法进行重写
def collate_fn(batch):
    '''
    #batch是list，其中是一个一个元组，每个元组是dataset中__getitem__的结果 即：(text, label)
    batch: [(text1, label1), (text2, label2), ...]  PS: 每一个text是一个[word1, word2, ...]
    :return: [text1, text2] & [label1, label2]
    '''
    # print('batch', batch)
    # zip(*) 表示解压
    text, label = list(zip(*batch)) # 将[(text1, label1), (text2, label2), ...]解压为[text1, text2] & [label1, label2]
    text = [ws.transform(words, max_len=MAX_LEN) for words in text]
    '''
        注意 input 和 label都需要是LongTensor类型
    '''
    # text, label = torch.tensor(text, dtype=torch.long), torch.tensor(label, dtype=torch.long)
    text, label = torch.LongTensor(text), torch.LongTensor(label)
    return text, label


def get_dataloader():
    dataset = ImdbDataset()
    dataloader = DataLoader(dataset=dataset,  batch_size=128, shuffle=True, collate_fn=collate_fn)
    return dataloader



if __name__ == '__main__':
    # dataset = ImdbDataset()
    # dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    # # print(dataset[0])
    #
    # # text = '''
    # # Bromwell High is a cartoon comedy. It ran at the same time as some other programs about school life, such as "Teachers". My 35 years in the teaching profession lead me to believe that Bromwell High's satire is much closer to reality than is "Teachers". The scramble to survive financially, the insightful students who can see right through their pathetic teachers' pomp, the pettiness of the whole situation, all remind me of the schools I knew and their students. When I saw the episode in which a student repeatedly tried to burn down the school, I immediately recalled ......... at .......... High. A classic line: INSPECTOR: I'm here to sack one of your teachers. STUDENT: Welcome to Bromwell High. I expect that many adults of my age think that Bromwell High is far fetched. What a pity that it isn't!
    # # '''
    # # print(tokenlize(text))
    #
    # for index, (text, label) in enumerate(dataloader):
    #     print('index', index)
    #     print('label', label)
    #     print('text', text)  # 如果直接使用dataloader加载数据，那么batch_size个句子中的单词会混合在一起
    #     break
    pass