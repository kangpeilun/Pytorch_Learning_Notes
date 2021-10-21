# Layer形状变换总结

## 1.nn.Linear(input, output)

第一个参数是传入nn.Linear层的数据，

第二个参数是nn.Linear层输出的数据

**注意input和output形状的变化**

```python
# 使用
self.fc = nn.Linear(input, output)

x = self.fc(x) # [batch_size, input_size]-->[batch_size, output_size]
```

## 2.nn.Embedding(num_embeddings,embedding_dim)

1. `num_embeddings`：词典的大小（整个数据集中不重复词语的个数）
2. `embedding_dim`：embedding的维度（用多长的一个向量来表示词语）

```python
# 使用
self.embedding = nn.Embedding(num_embeddings,embedding_dim)

x = self.embedding(x)  # [batch_size, seq_len]-->[batch_size, seq_len, embedding_dim]
# seq_len: 句子长度，即句子中词语的个数， 一般为一个人为规定的最大句子长度max_len
```

## 3.nn.LSTM(input_size,hidden_size,num_layers,batch_first,dropout,bidirectional)

1. `input_size `：输入数据的形状，即embedding_dim （词向量的维度）

2. `hidden_size`：隐藏层神经元的数量，即**每一层有多少个LSTM单元**

3. `num_layer` ：即RNN的中LSTM单元的层数

4. `batch_first`：默认值为False，输入的数据需要`[seq_len,batch,feature]`,如果为True，则为`[batch,seq_len,feature]`

5. `dropout`:dropout的比例，默认值为0。dropout是一种训练过程中让部分参数随机失活的一种方式，能够提高训练速度，同时能够解决过拟合的问题。这里是在LSTM的最后一层，对每个输出进行dropout

6. `bidirectional`：是否使用双向LSTM,默认是False

    **num_directions取值为1或2，1表示单向LSTM，2表示为双向LSTM**

```python
# 使用
self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first,dropout,bidirectional)

x = self.embedding(x)
x,(h_n, c_n) = self.lstm(x, (h_0, c_0))
# [batch_size, seq_len, embedding_dim]-->[batch_size, seq_len, num_directions * hidden_size]
'''
LSTM的输入为: batch_first=True
  input: [batch_size, seq_len, embedding_dim]   
  h_0:[num_layer*(1|2), batch_size, hidden_size]
  c_0:[num_layer*(1|2), batch_size, hidden_size]
'''

'''
LSTM的输出为: batch_first=True
	output: [batch_size, seq_len, num_directions * hidden_size]
	h_n: [batch_size, num_layers * num_directions, hidden_size]
	c_n: [batch_size, num_layers * num_directions, hidden_size]
'''
```



# Pytorch学习过程问题总结

## 1.RuntimeError: each element in list of batch should be of equal size

***PS：当Dataset中返回的input结果时字符串的时候，可以通过修改collate_fn解决***

出现问题的原因在于`Dataloader`中的参数`collate_fn`

`collate_fn`的默认值为torch自定义的`default_collate`,`collate_fn`的作用就是对每个batch进行处理，而默认的`default_collate`处理出错。

解决问题的思路：考虑自定义一个`collate_fn`，观察结果

 ```python
# collate_fn的作用是对每一个batch进行处理
# 因为torch默认的default_collate在处理文本时会出错，故这里对collate_fn方法进行重写
def collate_fn(batch):
    '''
    #batch是list，其中是一个一个元组，每个元组是dataset中__getitem__的结果 即：(text, label)
    batch: [(text, label), (text, label), ...]
    :return:
    '''
    print('batch', batch)
    text, label = list(zip(*batch))
    return text, label
  
  																															# ↓
dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
 ```

## 2.AttributeError: Can't get attribute 'word2seq' on <module '__main__' from 'E:/Pycharm/Pytorch学习/2.自然语言处理/utils/lib.py'>

**pickle要序列化的那个类所在的py文件，要保持干净，并能含有除了 class word2seq() 之外的其他内容**

https://blog.csdn.net/qq_42940160/article/details/120475329



## 3.TypeError: embedding(): argument 'indices' (position 2) must be Tensor, not list

传入模型的数据是一个list而不是一个tensor

**这时需要修改我们自定义的数据类中collate_fn的返回值即input和label的类型 为 LongTensor**

```python
def collate_fn(batch):
    '''
    #batch是list，其中是一个一个元组，每个元组是dataset中__getitem__的结果 即：(text, label)
    batch: [(text1, label1), (text2, label2), ...]  PS: 每一个text是一个[word1, word2, ...]
    :return: [text1, text2] & [label1, label2]
    '''
    print('batch', batch)
    text, label = list(zip(*batch)) # 将[(text1, label1), (text2, label2), ...]解压为[text1, text2] & [label1, label2]
    text = [ws.transform(words, max_len=MAX_LEN) for words in text]
    '''
        注意 input 和 label都需要是LongTensor类型
    '''
    text, label = torch.tensor(text, dtype=torch.long), torch.tensor(label, dtype=torch.long)
    return text, label
```

## 4.加载保存的模型时不需要再赋值

```python
'''
  导入模型是 模型的结果必须是实例化之后的那个变量 imdbmodel.load_state_dict 即 imdbmodel

  加载模型时 不需要再对实例赋一下值
  下面的写法是错误的
  imdbmodel = imdbmodel.load_state_dict(torch.load('./model/model.pkl'))
'''
    imdbmodel.load_state_dict(torch.load('./model/model.pkl'))
```

## 5.ValueError: dropout should be a number in range [0, 1] representing the probability of an element being zeroed

```python
dropout参数必须是一个[0,1]之间的数值
```

## 6.RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.

这是因为view()需要Tensor中的元素地址是连续的，但可能出现Tensor不连续的情况，所以先用 .contiguous() 将其在内存中变成连续分布：

```python
x = x.contiguous().view([-1, lib.max_len*lib.hidden_size])
```



# 编程技巧

## 1.self.count.get(word, 0) + 1 巧妙统计词频

```python
self.count.get(word, 0) + 1
如果当前字典中'word'存在则返回key对应的值并+1，如果'word'不存在则返回0+1
```

## 2.self.count = {word:value for word,value in self.count if value>min} 巧妙剔除不和要求的键值对

```python
self.count = {word:value for word,value in self.count if value>min} # 删除count中词频小于min的word
生产表达式，通过for循环遍历词典，并将不符合要求的键值对剔除掉
```

## 3.temp = sorted(self.count.items(), key=lambda x:x[-1], reverse=True)[:max_features] 根据value对dict排序

```python
'''
  sorted后会将元组变成列表
  self.count.items() 是一个可迭代对象, 其中的每一个值是一个(key,value)对
  key=lambda x:x[-1] 使字典中的key根据items中的value进行排序, x[-1]表示取最后一个值也就是value
  reverse=True 由大到小，降序排列
  [:max_features] 将排序后的前 max_features 个数取出来(因为sorted已经将dict_items变为list，故可以这样取值)
'''
# 这样得到的是一个列表，其中每个元素是一个二值元组
temp = sorted(self.count.items(), key=lambda x:x[-1], reverse=True)[:max_features] 
self.count = dict(temp)  # 将[(key, value), (key, value)] 转化为 {key:value, key:value}
```

## 4.创建word-->num的映射 巧妙对每个词语进行编号

```python
# 将 词语 和 编号对应起来
self.dict = {
  self.UNK_TAG: self.UNK,
  self.PAD_TAG: self.PAD
}

# 给每一个词语进行编号
for word in self.count:
'''
    因为原来的self.dict中已有self.UNK_TAG: self.UNK 和 self.PAD_TAG: self.PAD 两组键值对
    故新词的编号从 2 开始，也就不会和之前的重复
'''
  self.dict[word] = len(self.dict)
```

## 5.创建num-->word的映射 速度很快的转换

```python
self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))
# 直接取出原字典所有的values 和 keys，使用zip进行组合，才将其变成dict
```

## 6.pickel可以将整个class对象保存到文件中 加载pkl文件后能够继续使用该类

**PS：pickel序列化的类要保持“干净”，即该py文件中只能由该类，不能含有别的def方法**

```python
pickle.dump(ws, open('../model_data/ws.pkl', 'wb'))
# pickel可以将整个class对象保存到文件中，下次再使用时只需要将该文件读取成一个类，
# 那么便可以继续使用该文件对应的类中的数据和方法
ws = pickle.load(open('../model_data/ws.pkl', 'rb'))
# ws即为之前保存的对象
```

##7.字符串快速填充多个指定字符

```python
# 如果句子长度小于max_len则对句子填充max_len-len(text)个'PAD'
self.PAD_TAG = 'PAD'
text = text + [self.PAD_TAG] * (max_len - len(text))
```

## 8.获取output中每个预测值组成的list中最大元素及其对应索引位置

```python
'''
	tensor.max(dim=-1) 表示在行这一维度上取tensor的最大值，返回值为：最大值，对应索引位置
'''
pred = output.max(dim=-1)[-1]  # 取出预测结果中概率最大对应的索引值
```

## 9.torch.tensor判断output和label是否相等

```python
'''
    判断预测值是否和标签相等 pred.eq(label) 得到布尔值
    .float() 将布尔值转换为float浮点型
    .mean()  对整个tensor求平均值
'''
pred.eq(label).float().mean()
```

## 10.tqdm设置进度条

```python
# 导入tqdm时必须这样导入，否则会报TypeError: 'module' object is not callable的错误
from tqdm import tqdm
'''
	total=len(data_loader) 设置进度条的总长度
	ascii=True 设置进度条显示样式为 #####
	desc='测试：' 设置进度条提示信息
'''
tqdm(enumerate(data_loader), total=len(data_loader), ascii=True, desc='测试：')
```



## 11.端到端含义

**构建模型直接对原始数据进行处理，最终生成 相应的结果，该过程无须对数据进行别的转换**