import torch.nn as nn


class CNN_Zeng(nn.Module):
    def __init__(self, TFs_cell_line_pair=72):
        super(CNN_Zeng, self).__init__()
        """
            在PyTorch中，卷积、池化等的padding参数只允许一个或者两个
                当输入一个参数时，上下填充该数目的行，左右填充该数目的列
                当输入两个参数（A,B）时，左右均填充A列，上下均填充B行
            nn.ZeroPad2d让四侧填充不同数目的行或列
                （A,B,C,D）->(Left, Right, Top, Bottom)
        """
        """
            
        """
        self.Convolutions = nn.Sequential(
            nn.ZeroPad2d((11, 12, 0, 0)),
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4, 24)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=128)
        )
        self.GlobalMaxPool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.Dense = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(32, TFs_cell_line_pair),
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        """
            shape:
            [batch_size, 1, 4, 101] -> [batch_size, 128, 1, 101]
        """
        x = self.Convolutions(input)
        """
            shape:
            [batch_size, 128, 1, 101] -> [batch_size, 128, 1, 1]
        """
        x = self.GlobalMaxPool(x)
        """
            shape:
            [batch_size, 128, 1, 1] -> [batch_size, 128]
        """
        x = self.flatten(x)
        """
            shape:
            [batch_size, 128] -> [batch_size, NUM_CLASS]
        """
        output = self.Dense(x)
        return output