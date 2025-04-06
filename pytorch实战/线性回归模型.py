import torch
import numpy as np
import torch.nn as nn
from jinja2.optimizer import optimize

x_values=[i for i in range(15)]
x_trains=np.array(x_values, dtype=np.float32)
x_trains=x_trains.reshape(-1,1)   # -1表示自动计算行数，1表示列数
y_values=[i*2 + 1 for i in x_values]
y_trains=np.array(y_values, dtype=np.float32)
y_trains=y_trains.reshape(-1,1)
#导入数据

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
    def forward(self, x):
        out = self.linear(x)
        return out
input_dim=1
output_dim=1
model=LinearRegressionModel(input_dim, output_dim)

epochs=1000
learning_rate=0.01
optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion=nn.MSELoss()

# 训练模型
for epoch in range(epochs):
    inputs=torch.from_numpy(x_trains)
    targets=torch.from_numpy(y_trains)
    optimizer.zero_grad()  # 梯度清零
    outputs=model(inputs)  # 前向传播
    loss=criterion(outputs, targets)  # 计算损失
    loss.backward()    # 反向传播
    optimizer.step()  # 更新参数
    if (epoch+1)%50==0:
        print('epoch[{}/{}], loss:{:.4f}'.format(epoch+1, epochs, loss.item()))
