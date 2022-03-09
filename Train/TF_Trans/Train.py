import os

"""
    os.environ 是环境变量
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import torch.utils.data as Data
import torch.optim as optim

from NeuralNetwork.TF_Trans.model import make_model
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from EarlyStopping import EarlyStopping

from imblearn.over_sampling import SMOTE

class Batch:
    """
        To manege a batch of samples with mask in learning process
    """

    def __init__(self, src, pad=0):
        """
            src 是 model 的 input
            [batch_size, max_length, embedding(d_model)]
        """
        self.src = src
        """
            unsqueeze(-2) 在倒数第二个维度 插入一个维度
            e.g  [30, 10] --> [30, 1, 10]
            (src != pad) 如果scr中的元素和pad不相等 则src_mask这个位置是1 如果相等则是0
        """
        self.src_mask = (src != pad).unsqueeze(-2)


def make_dataset(TrainDataSet, ValidateDataSet, TFs_cell_line_pair, batch_size):
    """
        One-Hot Embedding
    """
    base_map = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]
    }
    TrainData = torch.randn(size=(TrainDataSet.shape[0], 101, 4))
    TrainLabel = torch.randn(size=(TrainDataSet.shape[0], TFs_cell_line_pair))
    for k in range(0, TrainDataSet.shape[0]):
        row = torch.randn(size=(101, 4))
        label = torch.zeros(size=(1, TFs_cell_line_pair))
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
        TensorDataset的目的是对Tensor打包 其通过每一个Tensor的第一个维度索引
        因此 传入的Tensor的第一维度必须相等
    """
    # Smote = SMOTE(random_state=25)
    # TrainData,  TrainLabel = Smote.fit_sample(TrainData, TrainLabel)
    TrainLoader = Data.DataLoader(dataset=torch.utils.data.TensorDataset(TrainData, TrainLabel),
                                  batch_size=batch_size, shuffle=True, num_workers=0)
    """
        Validation Dataset 构建过程同上
    """
    ValidateData = torch.randn(size=(ValidateDataSet.shape[0], 101, 4))
    ValidateLabel = torch.randn(size=(ValidateDataSet.shape[0], TFs_cell_line_pair))

    for k in range(0, ValidateDataSet.shape[0]):
        row = torch.randn(size=(101, 4))
        label = torch.zeros(size=(1, TFs_cell_line_pair))
        for location, base in enumerate(ValidateDataSet.loc[k, 1], start=0):
            row[location] = torch.as_tensor(base_map[base])
        ValidateData[k] = row
        label[0][ValidateDataSet.loc[k, 3]] = 1
        ValidateLabel[k] = label
    ValidateData = torch.transpose(ValidateData, 1, 2)
    ValidateLoader = Data.DataLoader(dataset=torch.utils.data.TensorDataset(ValidateData, ValidateLabel),
                                     shuffle=False, num_workers=0)
    return TrainLoader, ValidateLoader


def run(TrainLoader, ValidateLoader, Net, optimizer, loss_function, early_stopping, MAX_EPOCH, PATH):
    """
        Standard Training
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Net.to(device=device)
    for epoch in range(MAX_EPOCH):
        Net.train()
        ProgressBar = tqdm(TrainLoader)
        for data in ProgressBar:
            optimizer.zero_grad()

            ProgressBar.set_description("Epoch %d" % epoch)
            samples, label = data
            output = Net.forward(src=samples.to(device), src_mask=None)

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
            评估指标：Accuracy
        """
        acc = []
        valid_loss = []
        for valid_samples, valid_labels in ValidateLoader:
            """
                shape [batch_size, NUM_CLASS] 
            """
            valid_output = Net.forward(src=valid_samples.to(device), src_mask=None)
            valid_labels = valid_labels.to(device)

            valid_loss.append(loss_function(valid_output, valid_labels).item())
            """
                multi-class  评估
            """
            acc.append(accuracy_score(y_pred=torch.argmax(valid_output.cpu(), 1),
                                      y_true=torch.argmax(valid_labels.cpu(), 1)))
        valid_loss_avg = np.average(valid_loss)
        acc_avg = torch.mean(torch.Tensor(acc))

        early_stopping(valid_loss_avg, Net, path=PATH)

        print(f'\nValidate Accuracy: {acc_avg:.3f}\n')

        """
                保存模型
        """
    print('\n---Leaving  Train  Section---\n')


"""
    ***Main***
"""
"""
    Hyper parameters
"""
MAX_EPOCH = 20
BATCH_SIZE = 64
NUM_CLASS = 4
MS_num = 32
NConv = 3
D_model = MS_num * (2 ** (NConv - 1))
D_ff = D_model * 2
PATH = os.getcwd() + '/TF_Trans.pth'
"""
    load dataset
"""
TrainDataSet = pd.read_csv('SRFTrainDataset.csv', sep=' ', header=None)
ValidateDataSet = pd.read_csv('SRFTestDataset.data', sep=' ', header=None)

TrainLoader, ValidateLoader = make_dataset(TrainDataSet=TrainDataSet,
                                           ValidateDataSet=ValidateDataSet,
                                           TFs_cell_line_pair=NUM_CLASS,
                                           batch_size=BATCH_SIZE)

"""
    compile network
"""
Net = make_model(TFs_cell_line_pair=NUM_CLASS, NConv=NConv, NTrans=1,
                 ms_num=MS_num, d_model=D_model, d_ff=D_ff, h=8, dropout=0.1)
optimizer = optim.Adam(Net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0003)

criterion = nn.BCELoss()

early_stopping = EarlyStopping(patience=5, verbose=True)

"""
    Training start 
"""
run(TrainLoader=TrainLoader, ValidateLoader=ValidateLoader,
    Net=Net, optimizer=optimizer, loss_function=criterion,
    early_stopping=early_stopping, MAX_EPOCH=MAX_EPOCH, PATH=PATH)
