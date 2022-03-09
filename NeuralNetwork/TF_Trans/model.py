import torch.nn as nn
import copy
from convolutions import Convolution, MotifScanner
from attention import MultiHeadAttention
from otherlayer import PositionalEncoding, PositionwiseFeedForward
from encoderblock import Encoder, EncoderLayer
import torch


class TransformerEncoder(nn.Module):
    """
        注：
        DNA中各个核苷酸位点 按着自己的路径单独经过 Encoder中两个 layer
        Attention中这些路径间存在依赖关系  但是 FFN中不具有这样的依赖关系 因此各路径可在经过FFN时同时做运算
        在模型处理每个核苷酸时（DNA中的每个位置），Self-Attention让其可以查看input的DNA中的其它位置，以寻找思路来更好的对该位置Encode
    """
    """
        A standard Transformer-Encoder architecture
        Base for this and many other models.
    """

    def __init__(self, convolution, encoder, pos_embed, generator):
        super(TransformerEncoder, self).__init__()
        self.convolution = convolution
        self.encoder = encoder
        self.pos_embed = pos_embed
        self.generator = generator

    def forward(self, src, src_mask):
        """
            Take in and process masked src sequences
        """
        return self.generator(
            self.encode(self.convolution(src), src_mask))

    def encode(self, src, src_mask):
        return self.encoder(self.pos_embed(src), src_mask)


class Generator(nn.Module):
    """
        The output layer
    """

    def __init__(self, d_model, TFs_cell_line_pair):
        super(Generator, self).__init__()
        """
            此处还需要修改
        """
        self.proj = nn.Sequential(
            nn.Linear(d_model * 101, TFs_cell_line_pair),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
            x -> [batch_size, TFs_cell_line_pair]
        """
        return self.proj(torch.flatten(x, start_dim=1))


def make_model(TFs_cell_line_pair, NConv, NTrans, ms_num, d_model, d_ff, h, dropout=0.1):
    c = copy.deepcopy
    ms = MotifScanner(size=ms_num, dropout=dropout)
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = TransformerEncoder(convolution=Convolution(c(ms), N=NConv, dropout=dropout),
                               encoder=Encoder(layer=EncoderLayer(d_model, c(attn), c(ff), dropout), N=NTrans),
                               pos_embed=c(position),
                               generator=Generator(d_model=d_model, TFs_cell_line_pair=TFs_cell_line_pair))
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
