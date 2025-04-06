import random
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt

# 判断是否有GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 设置随机数种子
seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 定义生成器和判别器
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.sigmoid(self.map2(x))
        return self.map3(x)

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.elu(self.map2(x))
        x = torch.sigmoid(self.map3(x))
        return x

loss_function = nn.BCELoss()
learning_rate = 0.001
batch_size = 100
epochs = 100
input_size = 2  # 修正输入维度
hidden_size = 50
output_size = 2
generator = Generator(input_size, hidden_size, output_size).to(device)
discriminator = Discriminator(input_size, hidden_size, 1).to(device)

optimizer_generator = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=learning_rate)

def generate_real_data(batch_size):
    real_data = torch.randn(batch_size, input_size).to(device)
    return real_data

def generate_fake_data(generator, batch_size):
    noise = torch.randn(batch_size, input_size).to(device)
    fake_data = generator(noise)
    return fake_data

def train_discriminator(discriminator, optimizer, real_data, fake_data):
    N = real_data.size(0)
    real_labels = torch.ones(N, 1).to(device)
    fake_labels = torch.zeros(N, 1).to(device)
    optimizer.zero_grad()
    output_real = discriminator(real_data)
    loss_real = loss_function(output_real, real_labels)
    loss_real.backward()
    
    output_fake = discriminator(fake_data)
    loss_fake = loss_function(output_fake, fake_labels)
    loss_fake.backward()
    optimizer.step()
    return loss_real + loss_fake

def train_generator(generator, optimizer, fake_data):
    N = fake_data.size(0)
    real_labels = torch.ones(N, 1).to(device)
    optimizer.zero_grad()
    output = discriminator(fake_data)
    loss = loss_function(output, real_labels)
    loss.backward()
    optimizer.step()
    return loss

# 评估生成器
def evaluate_generator(generator, num_samples=1000):
    generator.eval()
    with torch.no_grad():
        fake_data = generate_fake_data(generator, num_samples)
        fake_data = fake_data.cpu().numpy()
        plt.figure(figsize=(10, 5))
        plt.scatter(fake_data[:, 0], fake_data[:, 1], label='Generated Data')
        plt.legend()
        plt.title('Generator Evaluation')
        plt.show()

# 评估判别器
def evaluate_discriminator(discriminator, real_data, fake_data):
    discriminator.eval()
    with torch.no_grad():
        real_output = discriminator(real_data).cpu().numpy()
        fake_output = discriminator(fake_data).cpu().numpy()
        plt.figure(figsize=(10, 5))
        plt.hist(real_output, bins=50, alpha=0.5, label='Real Data')
        plt.hist(fake_output, bins=50, alpha=0.5, label='Fake Data')
        plt.legend()
        plt.title('Discriminator Evaluation')
        plt.show()

for epoch in range(epochs):
    real_data = generate_real_data(batch_size)
    fake_data = generate_fake_data(generator, batch_size)
    loss_d = train_discriminator(discriminator, optimizer_discriminator, real_data, fake_data)
    fake_data = generate_fake_data(generator, batch_size)
    loss_g = train_generator(generator, optimizer_generator, fake_data)
    print('Epoch %d, Loss Discriminator: %.4f, Loss Generator: %.4f' % (epoch, loss_d.item(), loss_g.item()))

# 训练完成后评估生成器和判别器
evaluate_generator(generator)
evaluate_discriminator(discriminator, generate_real_data(1000), generate_fake_data(generator, 1000))
