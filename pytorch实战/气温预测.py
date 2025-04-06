import matplotlib.pyplot as plt
import numpy as np
import warnings
import datetime
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

features=pd.read_csv(r"D:\2345Downloads\神经网络实战分类与回归任务\temps.csv")
features.head()
years=features['year']
months=features['month']
days=features['day']
dates=[str(int(year))+'-'+str(int(month))+'-'+str(int(day)) for year,month,day in zip(years,months,days)]
dates=[datetime.datetime.strptime(date,'%Y-%m-%d') for date in dates]

plt.style.use('fivethirtyeight')   # 设置风格
# 设置布局
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize = (10,10))
fig.autofmt_xdate(rotation = 45)
# 标签值
ax1.plot(dates, features['actual'])
ax1.set_xlabel(''); ax1.set_ylabel('Temperature'); ax1.set_title('Max Temp')
# 昨天
ax2.plot(dates, features['temp_1'])
ax2.set_xlabel(''); ax2.set_ylabel('Temperature'); ax2.set_title('Previous Max Temp')
# 前天
ax3.plot(dates, features['temp_2'])
ax3.set_xlabel('Date'); ax3.set_ylabel('Temperature'); ax3.set_title('Two Days Prior Max Temp')
# 我的逗逼朋友
ax4.plot(dates, features['friend'])
ax4.set_xlabel('Date'); ax4.set_ylabel('Temperature'); ax4.set_title('Friend Estimate')
plt.tight_layout(pad=2)  # 设置间距
plt.show()

# 数据预处理
features = pd.get_dummies(features)   # 独热编码, 将特征值转换为独热编码,防止标准化影响
features.head(5)

labels=np.array(features['actual'])  # 标签列
features= features.drop('actual', axis = 1)  # 删除标签列

feature_list = list(features.columns)
features = np.array(features)
input_features=preprocessing.StandardScaler().fit_transform(features)

x=torch.tensor(input_features, dtype=torch.float)
y=torch.tensor(labels, dtype=torch.float)

weights=torch.randn((14,128),requires_grad=True, dtype=torch.float)
bias=torch.randn(128,requires_grad=True, dtype=torch.float)
weights2=torch.randn((128,1),requires_grad=True, dtype=torch.float)
bias2=torch.randn(1,requires_grad=True, dtype=torch.float)
lerrning_rate=0.001
losses=[]

for i in range(1000):
    hidden=torch.matmul(x,weights)+bias
    hidden=torch.relu(hidden)
    predictions=torch.matmul(hidden,weights2)+bias2
    loss=torch.mean((predictions-y)**2)
    losses.append(loss.data.numpy())

    if i%100==0:
        losses.append(loss)
        loss.backward()
        # 更新权重
        weights.data-=lerrning_rate*weights.grad.data
        bias.data-=lerrning_rate*bias.grad.data
        weights2.data-=lerrning_rate*weights2.grad.data
        bias2.data-=lerrning_rate*bias2.grad.data
        # 梯度清零
        weights.grad.data.zero_()
        bias.grad.data.zero_()
        weights2.grad.data.zero_()
        bias2.grad.data.zero_()
        print(loss)




