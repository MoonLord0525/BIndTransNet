import torch.nn as nn
from subutils import clones, LayerNorm, SublayerConnection

"""
    Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
"""


class Encoder(nn.Module):
    """
        The core of Encoder is a stack of N EncoderLayers
    """

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
            Pass the input (and mask) through each layer in turn
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


"""
    EncoderLayer(d_model, c(attn), c(ff), dropout)
"""


class EncoderLayer(nn.Module):
    """
        Encoder is made up of self-attn and feed forward
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        """
            A Encoder has 2 residual connection
        """
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        """
            self.size 即layer.size
        """
        self.size = size

    """
        EncoderLayer 有两个 block
        1. Multi-Head Attention + Add&Norm
        2. FeedForward + Add&Norm
    """

    def forward(self, x, mask):
        """
            lambda 表示匿名函数
            在本模型中，函数的传入参数是x
            匿名函数只能是一个表达式，不需要return
            函数return的是表示式的值
            本模型中，lambda x: self.self_attn(x, x, x, mask)的结果是 Multi-Head Attention计算得到的结果
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
