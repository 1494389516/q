import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import numpy as np
import matplotlib.pyplot as plt

# 数据准备
def generate_data(num_samples):
    X = np.random.randn(num_samples, 1)
    y = 6 * X + 3 + np.random.randn(num_samples, 1) * 0.1
    return X, y

# 新的生成器定义
class NewGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NewGenerator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 超参数
input_size = 1
hidden_size = 16
output_size = 1
num_epochs = 100
learning_rate = 0.001

# 数据生成
num_samples = 100
X, y = generate_data(num_samples)
X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32)

# 使用新的生成器
generator = NewGenerator(input_size, hidden_size, output_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(generator.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    generator.train()
    outputs = generator(X_train)
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
        predictions = model(X)
        plt.figure(figsize=(10, 5))
        plt.plot(y.numpy(), label='True')
        plt.plot(predictions.numpy(), label='Predicted')
        plt.legend()
        plt.title('Model Evaluation')
        plt.show()

# 训练完成后评估模型
evaluate_model(generator, X_train, y_train)
