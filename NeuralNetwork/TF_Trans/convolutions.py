import torch.nn as nn
import torch


class Convolution(nn.Module):
    def __init__(self, ms_layer, dropout, N):
        super(Convolution, self).__init__()
        self.layer_num = N

        self.conv_layers = nn.ModuleList()

        self.size = 0

        if self.layer_num == 1:
            self.ms_layer = ms_layer
            self.size = ms_layer.size
        else:
            self.ms_layer = ms_layer

            self.size = self.ms_layer.size * 2

            for i in range(self.layer_num - 1):
                self.conv_layers.append(ConvolutionLayer(size=self.size, dropout=dropout))
                self.size = self.size * 2
            self.size = self.size / 2

    def forward(self, x):
        """
            ms_layer:
            [batch_size, 4, seq_len] -> [batch_size, ms_layer.size, seq_len]
        """
        if self.layer_num == 1:
            x = self.ms_layer(x)
        else:
            x = self.ms_layer(x)
            for layer in self.conv_layers:
                x = layer(x)
        """
            [batch_size, seq_len, layer.size]
        """
        return torch.transpose(x, 1, 2)


class MotifScanner(nn.Module):
    """
        模体探测器
        该模块 网络中存在且只存在一个
    """

    def __init__(self, size, dropout):
        super(MotifScanner, self).__init__()
        """
            same卷积
        """
        self.conv = nn.Sequential(
            nn.ZeroPad2d((11, 12, 0, 0)),
            nn.Conv1d(in_channels=4, out_channels=size, kernel_size=24),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=size),
            nn.Dropout(p=dropout)
        )
        self.size = size

    def forward(self, x):
        return self.conv(x)


class ConvolutionLayer(nn.Module):
    def __init__(self, size, dropout):
        """
            size: 指的是convolution的 channel的 个数
        """
        super(ConvolutionLayer, self).__init__()
        """
            same卷积
        """
        """
            在CNN中不采取LayerNorm，LayerNorm在MLP和RNN中采取
            在CNN中采取BatchNorm，LayerNorm会让CNN无法收敛
        """
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=size // 2, out_channels=size, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=size),
            nn.Dropout(p=dropout)
        )
        self.size = size

    def forward(self, x):
        return self.conv(x)
