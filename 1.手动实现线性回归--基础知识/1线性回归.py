#-*- coding:utf-8 -*-
# author: Anefuer_kpl
# Email: 374774222@qq.com
# datatime: 2021-09-14 12:44
# project: Pytorch学习

import torch
import matplotlib.pyplot as plt


# 1.准备数据
# y=3x+0.8
x = torch.rand(500,1)
y_true = x*0.3 + 0.8

# 2.通过模型计算y_predict
w = torch.rand(1, requires_grad=True)  # 初始化参数
b = torch.tensor(0, requires_grad=True, dtype=torch.float32)

# 4.通过循环，反向传播，更新参数
learning_rate = 1e-3
for i in range(3000):
    # 3.计算loss
    y_predict = torch.matmul(x, w) + b
    loss = (y_true - y_predict).pow(2).mean()

    # 每次循环前判断梯度是否为0，如果不为0，则置为0
    if w.grad is not None:
        w.grad.data.zero_()
    if b.grad is not None:
        b.grad.data.zero_()

    # 反向传播
    loss.backward()
    w.data = w.data - learning_rate*w.grad
    b.data = b.data - learning_rate*b.grad
    print("w, b, loss",w.item(), b.item(), loss.item())


plt.figure()
plt.scatter(x.numpy().reshape(-1), y_true.numpy().reshape(-1))
y_predict = torch.matmul(x, w) + b
plt.plot(x.numpy().reshape(-1), y_predict.detach().numpy().reshape(-1), c='r')
plt.show()