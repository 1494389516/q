import torch.nn as nn
import numpy as np
from attr import validate
from torch.utils.data import TensorDataset, DataLoader
import torch
import  torch.nn.functional as F

# 数据生成
x_values = [i for i in range(15)]
y_values = [i * 2 + 1 for i in x_values]

# 转换为 NumPy 数组并调整形状
x_trains = np.array(x_values, dtype=np.float32).reshape(-1, 1)
y_trains = np.array(y_values, dtype=np.float32).reshape(-1, 1)

# 参数化配置
batch_size = 64
shuffle = True
bs=32
loss_func = F.cross_entropy
try:
    #创建训练数据集和数据加载器
 train_ds = TensorDataset(torch.from_numpy(x_trains), torch.from_numpy(y_trains))
 train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)

# 合理划分数据集，这里简单地将前80%作为训练集，后20%作为验证集
 split_idx = int(len(x_values) * 0.6)
 x_valid = np.array(x_values[split_idx:], dtype=np.float32).reshape(-1, 1)
 y_valid = np.array(y_values[split_idx:], dtype=np.float32).reshape(-1, 1)
 valid_ds = TensorDataset(torch.from_numpy(x_valid), torch.from_numpy(y_valid))
 valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

except Exception as e:
 print(f"数据处理过程中出现错误: {e}")

def get_data(train_ds, valid_ds,bs):
     return (
         DataLoader(train_ds, batch_size=bs, shuffle=True),
         DataLoader(valid_ds, batch_size=bs * 2),
      )

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
     for epoch in range(epochs):
         model.train()
         for xb, yb in train_dl:
             loss_batch(model, loss_func, xb, yb, opt)
             model.eval()
             with torch.no_grad():
                 loss_batch_results = [loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
                 losses, num = zip(*loss_batch_results)
                 val_losses=np.sum(np.multiply(losses,num))/np.sum(num)
                 print(f"epoch: {epoch}, '损失是': {val_losses}")

def get_model():
 model = nn.Linear(1, 1)
 return model, torch.optim.SGD(model.parameters(), lr=1e-4)
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)

train_dl1, valid_dl1 = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(25, model, loss_func, opt, train_dl1, valid_dl1)

