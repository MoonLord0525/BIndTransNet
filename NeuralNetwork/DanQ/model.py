import torch.nn as nn
import torch


class DanQ(nn.Module):
    def __init__(self, TFs_cell_line_pair=72):
        super(DanQ, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=320, kernel_size=26),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=320)
        )
        self.Pool = nn.MaxPool1d(kernel_size=19, stride=19)
        self.Drop1 = nn.Dropout(p=0.2)
        """
            BiLSTM batch first:
            注：
                num_directions 表示是否“双向”  Yes->2  No->1
            数据输入格式：
                input: [batch, seq_len, input_size]
                h0: [num_layers * num_directions, batch, hidden_size]
                c0: [num_layers * num_directions, batch, hidden_size]
            输出数据格式：
                output: [batch, seq_len, hidden_size * num_directions]
                hn: [num_layers * num_directions, batch, hidden_size]
                cn: [num_layers * num_directions, batch, hidden_size]
                注：
                    output保存 last layer 每个时间戳的输出h  
                    如果是双向LSTM 每个时间戳的输出h=[h正向, h逆向] (同一个时间戳的正向和逆向的h连接起来)
                    hn保存每个layer last时间戳的h 如果是双向LSTM 单独保存正向和逆向的last时间戳h
                    cn与hn一致 只是它保存的是c的值
        """
        self.BiLSTM = nn.LSTM(input_size=320, hidden_size=320, num_layers=2,
                              batch_first=True, dropout=0.5, bidirectional=True)
        self.flatten = nn.Flatten(start_dim=1)
        self.FullyConnection = nn.Sequential(
            nn.Linear(4*640, 925),
            nn.ReLU(),
            nn.Linear(925, TFs_cell_line_pair)
        )

    def forward(self, x):
        # input shape [batch_size, 4, 101]
        x = self.Conv(x)
        # convolution shape [batch_size, 320, 76]
        x = self.Pool(x)
        # pool shape [batch_size, 320, 4]
        x = self.Drop1(x)
        x_x = torch.transpose(x, 1, 2)
        # 转置 shape [batch_size, 4, 320]
        x, (hn, hc) = self.BiLSTM(x_x)
        # BiLSTM shape [batch_size, 4, 640]
        x = self.flatten(x)
        # view shape [batch_size, (4*640)]
        output = self.FullyConnection(x)
        # output shape [batch_size, TFs_cell_line_pair]
        return output
