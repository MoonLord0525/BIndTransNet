import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

"""
    主要包括 FeedForward-Network 和 Position-Encoding
"""


class PositionwiseFeedForward(nn.Module):
    """
        Implements FFN
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        """
            [d_model, d_ff]
        """
        self.w_1 = nn.Linear(d_model, d_ff)
        """
            [d_ff, d_model]
        """
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    """
        Implement PE function
    """

    def __init__(self, d_model, dropout, max_len=101):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        """
            在log空间中，计算一次 Positional Encodings
        """
        """
            pe -> [max_len, d_model]
            position -> [max_len, 1]
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        """
            div_term -> [d_model/2]
        """
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        """
            [0::2] -> [0, 2, 4, 6, ...]
            [1::2] -> [1, 3, 5, 7, ...]
        """
        """
            若 d_model =  512
            PE(1) = [sin(1/(10000^0/512)), cos(1/(10000^1/512)), sin(1/(10000^2/512)), cos(1/(10000^3/512)),...]
            pe[:, 0::2] 全部行取偶数列，修改每个元素的值
            pe[:, 1::2] 全部行取奇数列，修改每个元素的值
        """
        """
            position * div_term -> [max_length, d_model/2]
        """
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        """
            pe -> [1, max_len, d_model]
        """
        pe = pe.unsqueeze(0)
        """
            1. self.register_buffer 可以将Tensor register为 Buffer，在forward中使用self.mybuffer
            2. 定义parameter和buffer时都只需要传入Tensor即可。
            不需要将其转到GPU，当网络.cuda时候，会自己·将parameter和buffer转到指定的GPU
            3. 网络存储时也会将buffer存下，当网络load模型时，会将存储的模型的buffer赋值
            4. buffer的更新在forward中，optim.step只能更新nn.parameter类型的参数
        """
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)