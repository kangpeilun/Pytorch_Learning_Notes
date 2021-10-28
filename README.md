[Pytorch学习视频](https://www.bilibili.com/video/BV1CZ4y1w7mE?p=1)

### 自然语言处理常用标识符

```python
'''
    自然语言处理常用标识符:
        <UNK>: 低频词或未在词表中的词
        <PAD>: 补全字符
        <GO>/<SOS>: 句子起始标识符
        <EOS>: 句子结束标识符
        [SEP]：两个句子之间的分隔符
        [MASK]：填充被掩盖掉的字符
'''
```

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
3. `padding_idx`: （`必须为数字`）自然语言中使用批处理时候, 每个句子的长度并不一定是等长的, 这时候就需要对较短的句子进行padding, 填充的数据一般是0, 这个时候, 在进行词嵌入的时候就会进行相应的处理, nn.embedding会将填充的映射为0。**指定padding_idx后那么在计算梯度的时候就不会对该值进行梯度的更新，故会加快训练速度**以3 为例, 也就是说补长句子的时候是以3padding的, 这个时候我们液晶padding_idx设为3

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

PS: batch_first=True 只会影响output，不会影响h_n, c_n
    也就是说只有output的batch_size会被放在第一维
    h_n, c_n 的batch_size仍会被放在第二维

'''
LSTM的输入为: batch_first=True
  input: [batch_size, seq_len, embedding_dim]   
  h_0:[num_layer*(1|2), batch_size, hidden_size]
  c_0:[num_layer*(1|2), batch_size, hidden_size]
'''

'''
LSTM的输出为: batch_first=True
	output: [batch_size, seq_len, num_directions * hidden_size]
	h_n: [num_layers * num_directions, batch_size, hidden_size]
	c_n: [num_layers * num_directions, batch_size, hidden_size]
'''
```

## 4.nn.GRU()

GRU模块`torch.nn.GRU`，和LSTM的参数相同，含义相同，具体可参考文档

但是输入只剩下`gru(input,h_0)`，输出为`output, h_n`

其形状为：

`batch_first=True`

1. `output`:`(batch, seq_len, num_directions * hidden_size)`

2. `h_n`:`(batch, num_layers * num_directions, hidden_size)`

    ## 5.pack_padded_sequence(打包),pad_packed_sequence(解包) 

    `pack_padded_sequence`包的作用是**embedding后的结果打包**，然后传递给LSTM或GRU

    `pad_packed_sequence`包的作用是将**LSTM或GRU的output解包**

    这两个包可以加速LSTM或GRU的训练过程

    **注意两个包的输入和输入**

    ```python
    # 返回打包后的结果，形状和embedding之后的一样
    embeded = pack_padded_sequence(embeded, input_length, batch_first=True)
    
    # 返回解包之后的结果，以及每个batch中每个序列的长度
    output, output_length = pad_packed_sequence(output, batch_first=True, padding_value=config.num_sequence.PAD, total_length=config.max_len)
    ```

    ```python
    '''
    pack_padded_sequence中:
    	input_length 为输入句子的长度, 其形状要和batch进行匹配
    	
    pad_packed_sequence中:
    	padding_value 为解包时要还原的PAD，和nn.Embedding中的padding_idx意义相同
    	total_length=config.max_len  告诉pad_packed_sequence中句子最大长度是多少
    	返回值为 sentence, sentence_length
    		sentence： 句子
    		sentence： 每条句子本身的长度
    '''
    embeded = self.embedding(input) # [batch_size, max_len, embedding_dim] [128, 9, 100]
    embeded = pack_padded_sequence(embeded, input_length, batch_first=True)  # 把embedding后的结果打包
    output, hidden = self.gru(embeded)
    output, output_length = pad_packed_sequence(output, batch_first=True, padding_value=config.num_sequence.PAD, total_length=config.max_len)    # 把gru的输出解包 output:[batch_size, max_len, num_directions*hidden_size] [128, 9, 1*64]
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
    :return: text:[text1, text2, ...]  label:[label1, label2, ...]
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
    # text, label = torch.LongTensor(text), torch.LongTensor(label)
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



## 7.处理GitHub上的不允许100MB大文件上传

**删除掉commit中的大文件**

```python
# 1.在pycharm中底部的git标签页下的console中查看超过100MB的文件
通常会明确告诉你哪个文件超过了100MB

# 2.从已经commit的任务中移除掉大文件
git rm --cached <大文件路径>

# 3.提交操作
git commit --amend -CHEAD

之后重新push即可正常推送
```

## 8.pycharm 函数下输入三个“““回车后不出现参数，返回值等注释

```python
修改设置FIle-setting-Tools-Python Integrated Tools-Docstrings-Docstring formart,修改为自己想用的，推荐Epytext
```

![image-20211023170451014](https://raw.githubusercontent.com/kangpeilun/images/main/images/image-20211023170451014.png)

## 9.**batch_first=True只会影响output的结果**

```python
PS: batch_first=True 只会影响output，不会影响h_n, c_n
    也就是说只有output的batch_size会被放在第一维
    h_n, c_n 的batch_size仍会被放在第二维
```

## 10.ValueError: Expected target size (128, 14), got torch.Size([128, 10])

计算loss时一般遵循这样的规则：

​		**二维tensor和一维tensor进行计算**

​		三维tensor和二维tensor进行计算

```python
loss = F.nll_loss(decoder_outputs, label)
```

此例中decoder_outputs形状为[128, 10, 14]；label形状为[128, 10]

但因为未知的原因而无法计算

解决办法：`人为的将三维和二维的tensor降为二维和一维`

```python
decoder_outputs.view(decoder_outputs.size(0)*decoder_outputs.size(1), -1)
# 将decoder_outputs的第一个维度和第二个维度相乘，最后一个维度设为-1(表示自动适应)

label.view(-1)  # [batch_size*max_len]
# 因为label是二维的，直接设为-1(表示自动适应，即可自动变为一维)
```

```python
decoder_outputs = decoder_outputs.view(decoder_outputs.size(0)*decoder_outputs.size(1), -1)  
# [batch_size*max_len, vacab_size]
label = label.view(-1)  # [batch_size*max_len]
loss = F.nll_loss(decoder_outputs, label) # 计算loss
```

## 11.**Pytorch只要是自己造的数据在分配device是，都应该分配一遍**

应该 `.to(device)` 的有：

```python
1.实例化之后的模型
	seq2seq = Seq2Seq().to(config.device)
    
2.模型的input，label
    input = input.to(config.device)
    label = label.to(config.device)
    input_length = input_length.to(config.device)
    
3.创建模型过程中自己定义的用于存放数据的中间变量
'''
	这里的decoder_input、decoder_outputs均为NumDecoder模型中自定义的存放中间变量的tensor
'''
	decoder_input = torch.LongTensor([[config.num_sequence.SOS]]*config.train_batch_size).to(config.device)
    
	decoder_outputs = torch.zeros([config.train_batch_size, config.max_len+1, config.num_embedding]).to(config.device)
```

## 12.expected hidden size (1,256,64), got[1,32,64]

在使用pytorch的 LSTM (RNN) 构建多类文本分类网络时遇到此错误，网络结构没有问题，能够运行起来，但是运行到几个batch后就报错expected hidden size (1,256,64), got[1,32,64]

`原因`：

​		该错误是由于的训练数据不能被批量大小整除造成的。前面的batch都是256个，但是最后一个batch不足256，只有136个。

`解决办法`:

1.  修改batchsize，让数据集大小能整除batchsize

2.  `如果使用Dataloader，设置一个参数drop_last=True，会自动舍弃最后不足batchsize的batch`

    ## 13.ValueError: only one element tensors can be converted to Python scalars

    获取tensor中元素的值有两种方法

    1.  `tensor.item()` 当tensor中只有一个值时，可以使用该方法获取其中的值。`使用item()取多个值的话就会报上面的错误`

        ```python
        x = torch.randn(1)
        print(x)
        print(x.item())
         
        # 结果:
        tensor([-0.4464])
        -0.44643348455429077
        ```

    2.  `tensor.data` 当tensor中包含多个值时，可以使用该方法将所有的值都取出

        ```python
        x = torch.tensor([[-0.431], [0.2312]])
        print(x)
        print(x.data)
        
        # 结果:
        tensor([[-0.431], [0.2312]])
        tensor([[-0.431], [0.2312]])
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

## 7.字符串快速填充多个指定字符

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

###### 进阶用法

1.可以把tqdm当作一个可迭代对象

```python
bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), ascii=True)
```

2.针对这个对象可以设置更多个性化内容

```python
bar.set_description('train epoch:{}\tindex:{}\tloss:{:.3f}'.format(epoch, index, loss.item()))
```

完整代码

```python
bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), ascii=True)
    for index, (input, label, input_length, label_length) in bar:
        optimizer.zero_grad()  # 梯度置0
        decoder_outputs, _ = seq2seq(input, label, input_length, label_length)
        decoder_outputs = decoder_outputs.view(decoder_outputs.size(0)*decoder_outputs.size(1), -1)  # [batch_size*max_len, vacab_size]
        label = label.view(-1)  # [batch_size*max_len]
        loss = F.nll_loss(decoder_outputs, label) # 计算loss
        loss.backward()  # 反向传播
        optimizer.step() # 参数更新

        bar.set_description('train epoch:{}\tindex:{}\tloss:{:.3f}'.format(epoch, index, loss.item()))
        if index % 100 == 0:
            torch.save(seq2seq.state_dict(), config.model_save_path)
            torch.save(optimizer.state_dict(), config.optimizer_save_path)
```

## 11.端到端含义

**构建模型直接对原始数据进行处理，最终生成 相应的结果，该过程无须对数据进行别的转换**

## 12.巧妙划分train、test数据集比例

**如果数据是一条一条进行处理的，且希望在处理的过程中就对train、test的比例进行划分，可以使用这种方法划分比例**

```python
'''
    每次从该list中随机取一个，如果取出的值为1，则划为测试集
    这样就可以巧妙的控制划分的比例
    train/test = 3/1
    即有25%的数据作为测试集
'''
train_test_split = [0,0,0,1]

if random.choice(train_test_split) == 1:	# 如果随机选择的数据为1，则放入测试集；否则放入训练集
    test_file.write(line_cut + '\n')
else:
	train_file.write(line_cut + '\n')
```

## 13.torch.topk()快速获取tensor某一维度的前k个最大值及其索引

```python
torch.topk(decoder_output, 1) 获取tensor中最大的一个值, 默认从tensor的最后一维获取数据
    input：就是输入的tensor，也就是要取topk的张量
    k：就是取前k个最大的值。
    dim：就是在哪一维来取这k个值。
    lagest：默认是true表示取前k大的值，false则表示取前k小的值
    sorted：是否按照顺序输出，默认是true。

返回值:
    value: 最大值的数值
	index: 最大值所在索引位置
```

## 14.切片快速合并多个tensor

如果tensor1的形状为**[batch_sizem, max_len, vacab_len]**

​	   tensor2的形状为**[batch_size, vacab_len]**

那么就可以通过切片将tensor2合并到tensor1中

```python
for t in max_len:
    tensor1[:, t, :] = tensor2
```

## 15.np.random.seed(10)设置随机种子，防止数据集变化

为了防止每次重新进行训练之后，导致数据集发生变化，可以设置随机种子

```python
np.random.seed(10)  # 在seed()中填入种子数
```

## 16.**在使用Dataloader的collate_fn处理batch时，最好将每一组数据都转为torch.LongTensor类型**

否则在`.to(device)`时会出错

## 17.快速对list内容进行填充，变成所需形状

```python
# 1.在 列 方向上快速填充
[[SOS]]*batch_size
# 快速生成batch_size行[SOS]
结果：
	[
        [SOS],
        [SOS],
        [SOS],
        [SOS],
        [SOS],
        ......
    ]
    

# 2.在 行 方向上快速填充
[SOS]*batch_size
结果：
	[SOS, SOS, SOS, SOS, SOS, ......]
```

## 18.使用numpy快速判断两个list是否相等，并计算相等的个数

```python
'''
    使用numpy 将预测值 和 真实值 进行比较，并返回两个矩阵中对应元素是否相等的布尔值
    布尔值可以直接求和
'''
total_correct = sum(array1==array2)
```

```python
(array1 == array2) 返回两个矩阵中对应元素是否相等的逻辑值

(array1 == array2).all()当两个矩阵所有对应元素相等，返回一个逻辑值True

(array1 == array2).any()当两个矩阵所有任何一个对应元素相等，返回一个逻辑值True

>>> a = np.array([1,2,3])
>>> b = np.array([1,2,3])
>>> c = np.array([1,2,4])
>>> a == b
array([ True,  True,  True])
>>> a == c
array([ True,  True, False])
>>> (a == b).all()
True
>>> (a == c).all()
False
>>> (a == b).any()
True
>>> (a == c).any()
True
```

