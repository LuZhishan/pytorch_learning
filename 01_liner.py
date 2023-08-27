import torch
from torch import nn
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([3.0, 4.0])
true_b = 1.5
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


# 数据加载器，将输入数据按照batch_size分批次返回
def data_loader(data_arrays, batch_size, is_train=True):
    data_set = data.TensorDataset(*data_arrays)
    return data.DataLoader(data_set, batch_size, shuffle=is_train)


batch_size = 100
data_iter = data_loader((features, labels), batch_size)

# 构建网络模型
net = nn.Sequential(nn.Linear(2, 1))  # 添加一个线性层
net[0].weight.data.normal_(0, 0.01)  # 给权重赋值
net[0].bias.data.fill_(0)
loss = nn.MSELoss()  # 设置损失函数
trainer = torch.optim.SGD(net.parameters(), lr=0.03)  # 设置优化函数

# 开始训练
num_epochs = 10
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()  # 梯度清零
        l.backward()  # 反向传递
        trainer.step()  # 数据更新
    l = loss(net(features), labels)
    print(f"epoch {epoch + 1}, loss {l:f}")
