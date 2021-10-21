# -*- coding:utf-8 -*-
# author: Anefuer_kpl
# Email: 374774222@qq.com
# datatime: 2021-09-22 17:25
# project: Pytorch学习

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

BATCH_SIZE = 256
EPOCH = 3

'''
    切记：torch中处理图片需要将shape `(H,W,C)` 改为 `(C,H,W)`, 也就是将通道数放到最前面
         使用 `torchvision.transforms.ToTensor` 进行转换
'''


# 1.准备数据集
def get_dataloader(train=True):
    transform_fn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.1307, 0.3081)  # mean 和 std
    ])
    mnist = MNIST(root='./data',
                  train=train,
                  download=True,
                  transform=transform_fn)  # PS: 如果root目录下有数据，那么就不会再下载了

    data_loader = DataLoader(dataset=mnist, batch_size=BATCH_SIZE, shuffle=True)
    return data_loader, len(mnist)


# 2.构建模型
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.fc1 = nn.Linear(1 * 28 * 28, 28)
        self.fc2 = nn.Linear(28, 10)

    def forward(self, input):
        '''
        :param input: [batch_size, 1, 28, 28]
        :return:
        '''
        # 1.修改形状
        x = input.view([-1, 1 * 28 * 28])
        # 2.进行全连接操作
        x = self.fc1(x)
        '''
          3.激活函数处理，形状不会发生变化
        '''
        x = F.relu(x)
        # 4.再次进行全连接操作
        out = self.fc2(x)

        return F.log_softmax(out)


mnist_net = MnistModel()
optimizer = optim.Adam(mnist_net.parameters(), lr=1e-3)
# 3.进行训练
def train(epoch):
    data_loader, length = get_dataloader()
    for index, (input, label) in enumerate(data_loader):
        optimizer.zero_grad()
        output = mnist_net(input)  # 调用模型，得到预测值
        loss = F.nll_loss(output, label)  # 带权的交叉熵损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        if index % 100 == 0:
            # 4.模型保存
            # 每隔100次保存一次
            torch.save(mnist_net.state_dict(), f'model/mnist/mnist_net_index{index}_epoch{epoch}.pt')  # 保存模型参数
            torch.save(optimizer.state_dict(), f'model/mnist/mnist_opt_index{index}_epoch{epoch}.pt')  # 保存优化器参数
            print(f'epoch({epoch}) loss:', loss.item())


# 4.进行测试
def test():
    test_loss = 0
    acc = 0
    mnist_net.eval()  # 设置模型为评估模式
    test_dataloader, length = get_dataloader(train=False)
    with torch.no_grad(): # 不计算梯度
        for input, label in test_dataloader:
            output = mnist_net(input)
            # print('output', output)
            test_loss += F.nll_loss(output, label, reduction='sum').item()
            '''
                损失函数nll_loss讲解：https://www.gentlecp.com/articles/874.html 
                reduction='sum' 表示对一个batch的loss求和， 而 test_loss+= 表示对所有batch的loss进行求和
            '''
            predict = output.data.max(1, keepdim=True)[1]
            # print('predict', predict)
            '''
                找到输出中 概率最大的那一项 对应的索引（因为索引正好对应着标签）， 
                keepdim=True表示保留最大值对应的索引
            '''
            acc += predict.eq(label.data.view_as(predict)).sum()  # 对预测正确的个数累加
            '''
                view_as返回被视作与给定的tensor相同大小的原tensor
                一个label中包含了一个batch中所有的标签，view_as把label变成和predict一样的形状
                使用 eq 对两个tensor中的值进行比较，如：
                    predict     [0, 3, 4, 1]
                    label       [2, 3, 4, 2]
                返回值是bool列表，然后用sum对bool列表进行求和，从而计算出该batch中有多少个是预测正确的
                acc+= 表示对所有的batch预测正确的个数进行求和
            '''

    test_loss /= length  # 计算平均损失, length 为mnist数据集的数量
    print('Test set: Avg. loss: {:.4f}, acc:{}/{} {:.2f}%'.format(
        test_loss, acc, length, 100.*acc/length
    ))

if __name__ == '__main__':
    # for i in range(EPOCH):
    #     train(i)

    # 加载训练好的模型
    mnist_net.load_state_dict(torch.load('model/mnist/mnist_net_index400_epoch2.pt'))
    optimizer.load_state_dict(torch.load('model/mnist/mnist_opt_index400_epoch2.pt'))
    test()