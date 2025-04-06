import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 数据准备
def generate_data(seq_length, num_samples):
    X = np.random.randn(num_samples, seq_length, 1)
    y = np.sum(X, axis=1)
    return X, y

# 模型定义
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True).to(device)
        self.fc = nn.Linear(hidden_size, output_size).to(device)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# 超参数
input_size = 1
hidden_size = 16
output_size = 1
num_epochs = 100
learning_rate = 0.001

# 数据生成
seq_length = 10
num_samples = 1000
X, y = generate_data(seq_length, num_samples)
X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32)

# 模型实例化
model = RNNModel(input_size, hidden_size, output_size).to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)   
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 模型评估
def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        predictions = model(X.to(device))
        plt.figure(figsize=(10, 5))
        plt.plot(y.cpu().numpy(), label='True')
        plt.plot(predictions.cpu().numpy(), label='Predicted')
        plt.legend()
        plt.title('Model Evaluation')
        plt.show()

# 训练完成后评估模型
evaluate_model(model, X_train, y_train)



