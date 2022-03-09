import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from subutils import clones

"""
    Scaled Dot product attention
"""


def attention(query, key, value, mask=None, dropout=None):
    """
        如果 h=8，d_model=512
        shape: query=key=value ---->[batch_size, 8, max_length, 64]
    """
    d_k = query.size(-1)
    """
        K的维度交换后为：[batch_size, 8, 64, max_length]
        score的维度为：[batch_size, 8, max_length, max_length]
    """
    """
        matmul 是张量的乘法 实际上是倒数第一、二维的乘法
        [batch_size, 8, max_length, 64] × [batch_size, 8, 64, max_length] - > [batch_size, 8, max_length, max_length]
    """
    score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    """
        padding mask
    """
    if mask is not None:
        """
            mask是一个Tensor，其中的值均是1或0
            mask==0，mask中是0的记作True，mask中是1的记作False
            masked_fill 将score中位置在mask中是True的地方，其中的value拿-1e9替换（相当于拿零替换）
        """
        score = score.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(score, dim=-1)
    if dropout is not None:
        """
            p_attn: [batch_size, 8, max_length, max_length]
        """
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


"""
    Multi-Head Attention
"""


class MultiHeadAttention(nn.Module):
    """
        Take in number of heads and model size
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        """
            assert condition
            即
            if not condition:
                raise AssertionError()
        """
        assert d_model % h == 0
        """
            // 整数除法
            /  浮点数除法
        """
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
            shape: query=key=value ---->: [batch_size, max_length, embedding_dim]
        """
        if mask is not None:
            """
                same mask applied to all h heads
            """
            mask = mask.unsqueeze(1)
            print(mask.shape)
        nbatches = query.size(0)

        """
            第一步：将q,k,v分别与Wq,Wk,Wv矩阵进行相乘
            shape: Wq=Wk=Wv----->[embedding_dim, embedding_dim]
            第二步：将获得Q、K、V在第三个维度上切分
            shape: [batch_size, max_length, h, d_k]
            第三步：交换第二和第三维度
            shape: [batch_size, h, max_length, d_k]
        """
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        """
            zip以元组的形式获得 (linear[0], query), (linear[1], key), (linear[2], value)
            l(x)将得到 query = linear[0](query), key = linear[1](key), value = linear[2](value)
            view函数 将 query, key, value 在第三个维度上切割，即
            shape of query = [batch_size, max_length, h, d_k]
            shape of key = [batch_size, max_length, h, d_k]
            shape of value = [batch_size, max_length, h, d_k]
            Transpose: query , key , value 's shape -> [batch_size, h, max_length, d_k]
            
            快速将列表的值赋给多个变量,
            query, key, value = [query, key, value]
        """

        """
            经过 attention 维度不变，shape: [batch_size, 8, max_length, 64]
            self.attn 是 weight maps
        """
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        """
            将维度进行还原
            [batch_size, max_length, h, d_k] -> [batch_size, max_length, d_model]
        """
        """
            contiguous函数将 张量 在内存中的存储变得连续
            view 只能在contiguous的variable上
        """
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        """
            与W0大矩阵相乘
        """
        return self.linears[-1](x)
