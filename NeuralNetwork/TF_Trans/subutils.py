import torch
import torch.nn as nn
import copy


def clones(module, N):
    """
        clones函数：Produce N layers
    """
    """
        nn.ModuleList 是 容器(containers)， 我们可以添加 模块(module)到其中
        可以把nn.Module的子类 (nn.Conv2d, nn.Linear 等)添加到 nn.ModuleList 这个 list 中
        添加的方法和Python自带的list基本相同， 无非是append，extend等操作
        nn.ModuleList 不等同于传统的 list，nn.ModuleList中的 module 会注册到神经网络中
        注意，nn.ModuleList没有定义一个网络，只是将不同模块存储在一起，这些模块间没有次序
        nn.ModuleList中的 module 可被 call 多次，但 parameter 共享
    """
    """
        copy.deepcopy是硬拷贝函数
        硬拷贝（深拷贝）：改变以前的对象，不影响现在的对象
    """
    return nn.ModuleList(copy.deepcopy(module) for _ in range(N))


class LayerNorm(nn.Module):
    """
        Construct a layernorm module
        LayerNorm：取同一个样本的不同通道归一化
        BatchNorm：取不同样本的同一个通道归一化
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        """
            nn.Parameter 是继承自 Tensor 的子类，
            nn.Parameter 将被认为是Trainable的 parameter，添加到 parameter()这个迭代器中
            即存在于 net.parameter()中
        """
        """
            features = layer.size = dimension of embedding
        """
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        """
            dim = -1 反向索引
            keepdim 保持维度不变
        """
        """
            [batch, length, embedding]
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        """
            mean, std -> [batch, length, 1]
        """
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
        The residual connection
        For code simplicity, layer norm is first as opposed to last
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
            Apply residual connection to any sublayer with same size
            sublayer 即 FeedForward 或 Multi-Head Attention
        """
        """
            layer norm -> sublayer -> dropout -> add
        """
        return x + self.dropout(sublayer(self.norm(x)))
