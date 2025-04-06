import matplotlib.pyplot as plt
import numpy as np
import warnings
import datetime
import pandas as pd
import torch
from sklearn import preprocessing

features=pd.read_csv(r"D:\2345Downloads\神经网络实战分类与回归任务\temps.csv")
# 数据预处理
features = pd.get_dummies(features)   # 独热编码, 将特征值转换为独热编码,防止标准化影响
features.head(5)

labels=np.array(features['actual'])  # 标签列
features= features.drop('actual', axis = 1)  # 删除标签列

feature_list = list(features.columns)
features = np.array(features)
input_features=preprocessing.StandardScaler().fit_transform(features)
input_size=input_features.shape[1]
hidden_size=128
output_size=1
batch_size=16

learning_rate=0.001
losses=[]
my__nn=torch.nn.Sequential(
    torch.nn.Linear(input_size,hidden_size),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size,output_size),
)
cost=torch.nn.MSELoss()
optimizer=torch.optim.Adam(my__nn.parameters(),lr=learning_rate)
for i in range(1000):
    batch_loss=[]
    for j in range(0,len(input_features),batch_size):
        end=min(j+batch_size,len(input_features))
        xx=torch.tensor(input_features[j:end],dtype=torch.float32,requires_grad=True)
        yy=torch.tensor(labels[j:end],dtype=torch.float32,requires_grad=True)
        y_pred=my__nn(xx)
        loss=cost(y_pred,yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
    if i%100==0:
        print("epoch:",i,"loss:",np.mean(batch_loss))
        losses.append(np.mean(batch_loss))


