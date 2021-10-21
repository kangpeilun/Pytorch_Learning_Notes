#-*- coding:utf-8 -*-
# author: Anefuer_kpl
# Email: 374774222@qq.com
# datatime: 2021-10-18 11:38
# project: Pytorch学习

'''准备问答数据'''

import pandas as pd
import config
import os

'''
    在内部的py文件，如果想要调用别的路径下的文件，必须使用相对路径
'''
answer_path = config.answer_path
question_path = config.question_path

def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def is_file_right():
    # 判断csv文件是否符合要求
    answers = pd.read_csv(answer_path)
    column_list = answers.columns  # 获取csv文件的标签
    # print(column_list)
    assert 'question_id' in column_list and 'content' in column_list, 'answer.csv文件必须包含问题id和答案'

    questions = pd.read_csv(question_path) # 指定 question_id 列为索引
    column_list = questions.columns
    assert 'question_id' in column_list and 'content' in column_list, 'question.csv文件必须包含问题id和答案'

    return questions, answers


def create_QA_data():
    '''
    创建QA数据集
    格式：[question, answer]
    '''
    QA_data = '../corpus/QA_data/QA_data.csv'
    questions, answers = is_file_right()
    temp_questions = []
    temp_answers = []
    for question_id,question_content in zip(questions['question_id'], questions['content']):
        '''
            根据question.csv中的question_id锁定answer.csv对应的答案
            answers[answers['question_id']==question_id]['content']  锁定对应的answer.csv中的答案，并取出对应的content值
            .tolist()[0] 将panda对象转化为list，并去掉空格
        '''
        try:
            answer_content = answers[answers['question_id']==question_id]['content'].tolist()[0]
            temp_questions.append(question_content)
            temp_answers.append(answer_content)
            print(question_content, '\t',answer_content,)
        except:  # 如果在answer.csv找不到对应的答案，则跳过该行问题
            continue

    pd.DataFrame({'question':temp_questions, 'answer':temp_answers}).to_csv(QA_data, index=False)  # 将整理后的数据写入csv文件
    print('Finish Create data!')


if __name__ == '__main__':
    create_QA_data()

    # data = pd.read_csv('../corpus/QA_data/QA_data.csv')
    # print(data['question'][0],data['answer'][0])