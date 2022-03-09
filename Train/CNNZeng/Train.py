import os

"""
    os.environ 是环境变量
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import numpy as np

from NeuralNetwork.CNNZeng.model import CNN_Zeng
from EarlyStopping import EarlyStopping
from tqdm import tqdm
from sklearn.metrics import accuracy_score

"""
    Hyper Parameters
    注：
    将模型保存在当前文件夹下
"""
MAX_EPOCH = 40
BATCH_SIZE = 100
MOMENTUM = 0.99
DELTA = 1e-6
NUM_CLASS = 4
PATH = os.getcwd() + '/CNNZeng.pth'

"""
    GPU device
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
    load data set
"""
base_map = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1],
    'N': [0, 0, 0, 0]
}

"""
    TrainData[1] -> DNA sequence
    TrainData[3] -> label
"""
TrainDataSet = pd.read_csv('TrainData.csv', sep=' ', header=None)
"""
        TrainData.shape[0] -> 行数
"""
TrainData = torch.randn(size=(TrainDataSet.shape[0], 101, 4))
TrainLabel = torch.randn(size=(TrainDataSet.shape[0], NUM_CLASS))
for k in range(0, TrainDataSet.shape[0]):
    row = torch.randn(size=(101, 4))
    label = torch.zeros(size=(1, NUM_CLASS))
    """
        base表示碱基的类型（A,T,C,G） location表示碱基在DNA中的位置
    """
    for location, base in enumerate(TrainDataSet.loc[k, 1], start=0):
        """
            torch.as_tensor 将任意的数据类型转换至Tensor
        """
        row[location] = torch.as_tensor(base_map[base])
    TrainData[k] = row
    """
        label vector 中相关位置 置1
    """
    label[0][TrainDataSet.loc[k, 3]] = 1
    TrainLabel[k] = label
"""
    shape [batch_size, 4, 101]
"""
TrainData = torch.transpose(TrainData, 1, 2)
"""
    shape [batch_size, 1, 4, 101]
"""
TrainData = TrainData.unsqueeze(dim=1)
"""
    TensorDataset的目的是对Tensor打包 其通过每一个Tensor的第一个维度索引
    因此 传入的Tensor的第一维度必须相等
"""
TrainLoader = Data.DataLoader(dataset=torch.utils.data.TensorDataset(TrainData, TrainLabel),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
"""
    Validation Dataset 构建过程同上
"""
ValidateDataSet = pd.read_csv('ValidateData.csv', sep=' ', header=None)

ValidateData = torch.randn(size=(ValidateDataSet.shape[0], 101, 4))
ValidateLabel = torch.randn(size=(ValidateDataSet.shape[0], NUM_CLASS))

for k in range(0, ValidateDataSet.shape[0]):
    row = torch.randn(size=(101, 4))
    label = torch.zeros(size=(1, NUM_CLASS))
    for location, base in enumerate(ValidateDataSet.loc[k, 1], start=0):
        row[location] = torch.as_tensor(base_map[base])
    ValidateData[k] = row
    label[0][ValidateDataSet.loc[k, 3]] = 1
    ValidateLabel[k] = label
ValidateData = torch.transpose(ValidateData, 1, 2)
ValidateData = ValidateData.unsqueeze(dim=1)
ValidateLoader = Data.DataLoader(dataset=torch.utils.data.TensorDataset(ValidateData, ValidateLabel),
                                 shuffle=True, num_workers=0)

"""
    compile network
"""
Net = CNN_Zeng(TFs_cell_line_pair=NUM_CLASS)
Net.to(device=device)
"""
    ADADELTA: An Adaptive Learning Rate Method
"""
optimizer = optim.Adadelta(Net.parameters(), rho=MOMENTUM, eps=DELTA)
"""
    当验证集损失在连续 patient个 Epoch中都没有得到降低时， 停止模型的Train， 防止过拟合
"""
early_stopping = EarlyStopping(patience=5, verbose=True)

loss_function = nn.BCELoss()

"""
    Training start
"""
for epoch in range(MAX_EPOCH):
    Net.train()
    ProgressBar = tqdm(TrainLoader)
    for data in ProgressBar:
        optimizer.zero_grad()

        ProgressBar.set_description("Epoch %d" % epoch)
        samples, label = data
        output = Net(samples.to(device))

        loss = loss_function(output, label.to(device))
        """
            实时显示loss
        """
        ProgressBar.set_postfix(loss=loss.item())

        loss.backward()
        optimizer.step()
    """
        Validate
    """
    Net.eval()
    """
        学习率衰减时需要 valid_loss
    """
    valid_loss = []
    """
        评估指标：Accuracy
    """
    acc = []
    with torch.no_grad():
        for valid_samples, valid_labels in ValidateLoader:
            """
                shape [batch_size, NUM_CLASS] 
            """
            valid_output = Net(valid_samples.to(device))
            valid_labels = valid_labels.to(device)

            valid_loss.append(loss_function(valid_output, valid_labels).item())
            """
                multi-class  评估
                注：valid_output 是 requires grad 因此需要 .detach().numpy()
            """
            acc.append(accuracy_score(y_pred=torch.argmax(valid_output, 1),
                                      y_true=torch.argmax(valid_labels, 1)))
        valid_loss_avg = np.average(valid_loss)
        acc_avg = np.average(acc)
        """
            早停
            保存检查点
        """
        early_stopping(valid_loss_avg, Net, path=PATH)

        print(f'Validate Accuracy: {acc_avg:.3f}')

print('\n---Leaving  Train  Section---\n')