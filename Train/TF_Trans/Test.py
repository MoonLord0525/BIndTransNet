import os
import torch
import pandas as pd
import numpy as np
import torch.utils.data as Data

from NeuralNetwork.TF_Trans.model import make_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import cohen_kappa_score


"""
    Hyper Parameters
"""
NUM_CLASS = 4
PATH = os.getcwd() + '/TF_Trans.pth'

"""
    python可变参数：
    *args 表示任意数量的无名参数，它是一个Tuple
    **kwargs 表示关键字参数，它是一个Dictionary
"""
params = {'batch_size': 1, 'num_workers': 0}

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

TestDataSet = pd.read_csv('TestData.csv', sep=' ', header=None)

TestData = torch.randn(size=(TestDataSet.shape[0], 101, 4))
TestLabel = torch.randn(size=(TestDataSet.shape[0], NUM_CLASS))

for k in range(0, TestDataSet.shape[0]):
    row = torch.randn(size=(101, 4))
    label = torch.zeros(size=(1, NUM_CLASS))
    for location, base in enumerate(TestDataSet.loc[k, 1], start=0):
        row[location] = torch.as_tensor(base_map[base])
    TestData[k] = row
    label[0][TestDataSet.loc[k, 3]] = 1
    TestLabel[k] = label
TestData = torch.transpose(TestData, 1, 2)
TestLoader = Data.DataLoader(dataset=torch.utils.data.TensorDataset(TestData, TestLabel),
                             **params)

"""
    compile network
"""
Net = make_model(TFs_cell_line_pair=NUM_CLASS, NConv=1, NTrans=1,
                 d_model=32, d_ff=64, h=8, dropout=0.1)
Net.load_state_dict(torch.load(PATH))

"""
    Test start
"""
Net.eval()
predicted_value = []
ground_label = []
for sample, label in TestLoader:
    output = Net.forward(src=sample, src_mask=None)
    predicted_value.append(np.argmax(output.detach().numpy(), 1).item())
    ground_label.append(np.argmax(label, 1).item())

"""
    评估指标：
    Accuracy    JaccardScore    Kappa
    注：
    当Accuracy是1时 Kappa是nan
"""
acc = accuracy_score(y_pred=predicted_value, y_true=ground_label)
j_score = jaccard_score(y_pred=predicted_value, y_true=ground_label)
kappa = cohen_kappa_score(y1=predicted_value, y2=ground_label, labels=None, weights='quadratic')

print(f'Accuracy: {acc:.3f} ' + f'JaccardScore: {j_score:.3f}' + f'Kappa: {kappa:.3f}')

print('\n---Leaving  Test  Section---\n')