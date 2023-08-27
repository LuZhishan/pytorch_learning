import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 数据加载器，将输入数据按照batch_size分批次返回
def data_loader(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="data", train=True, download=True, transform=trans
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root="data", train=False, download=True, transform=trans
    )
    return (
        data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
        data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4),
    )


# 计算分类精度
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


# 初始化权重
def init_wights(m):
    if type(m) == nn.Linear:  # 只初始化线性层
        nn.init.normal_(m.weight, mean=0, std=0.01)


# 构建网络模型，一个展平层，一个线性层
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
net.apply(init_wights)
loss = nn.CrossEntropyLoss(reduction="none")
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

net.train()  # 切换到训练模式
# net.eval() 切换到评估模式
metric = Accumulator(3)  # 训练损失总和、训练准确度总和、样本数
train_iter, test_iter = data_loader(32)
num_epochs = 10
for epoch in range(num_epochs):
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        trainer.zero_grad()
        l.mean().backward()
        trainer.step()
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    print(f"训练损失: {metric[0] / metric[2]}, 训练精度: {metric[1] / metric[2]}")
