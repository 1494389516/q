from typing import dataclass_transform
import os
from torch.utils.data import DataLoader, dataset
from torchvision import transforms
from torchvision import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

data_dir=r"D:\BaiduNetdiskDownload\flower_data"
train_dir=r"D:\BaiduNetdiskDownload\flower_data\train"
valid_dir=r"D:\BaiduNetdiskDownload\flower_data\valid"

data_transforms={
    'train':transforms.Compose([
        transforms.RandomResizedCrop(45),
        transforms.RandomHorizontalFlip(p=0.6),  # 随机水平翻转
        transforms.RandomVerticalFlip(p=0.6),  # 随机垂直翻转
         transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    'valid':transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
}
batch_size=16# 设置batch_size
image_datasets={x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ['train','valid']}
dataloaders={x:torch.utils.data.DataLoader(image_datasets[x],batch_size=batch_size,shuffle=True,num_workers=4) for x in ['train','valid']}
#dataloader = DataLoader(dataset, batch_size=32, num_workers=0)
data_size={x:len(image_datasets[x]) for x in ['train','valid']}
class_names=image_datasets['train'].classes

# 定义一个简单的卷积神经网络模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 11 * 11, 512)  # 假设输入图像是45x45
        self.fc2 = nn.Linear(512, len(class_names))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    # 初始化模型、损失函数和优化器
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_steps = 20
    for step in range(num_steps):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloaders['train']:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        step_loss = running_loss / data_size['train']
        print(f'Step: {step+1}/{num_steps}, Loss: {step_loss:.4f}%')