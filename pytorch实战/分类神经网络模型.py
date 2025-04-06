import torch.nn.functional as F
import torch.nn as nn
import torch
from pytorch实战.线性回归模型 import x_trains
from pytorch实战.线性回归模型 import y_trains

loss_func=nn.CrossEntropyLoss()
bs=64
xb=x_trains[0:bs]
yb=y_trains[0:bs]
weights=torch.randn(784,10)
bias=torch.zeros(10, requires_grad=True)
def mobel():
    return xb@weights+bias
