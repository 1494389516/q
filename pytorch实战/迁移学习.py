import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models  # 添加 models 的导入

train_dir = r"D:\BaiduNetdiskDownload\flower_data\train"
valid_dir = r"D:\BaiduNetdiskDownload\flower_data\valid"
model_name = 'resnet18'  # 修正拼写错误
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 修正拼写错误

# 数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 加载数据集
image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
}

# 创建数据加载器
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
    'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32, shuffle=False, num_workers=4)
}

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18 """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)
        input_size = 224

    return model_ft, input_size  # 返回模型和输入大小

# 定义并初始化模型
model_name = "resnet"
feature_extract = True
model, input_size = initialize_model(model_name, feature_extract, use_pretrained=True)
optimizer = optim.Adam(model.parameters(), lr=0.01)  # 移除 momentum 参数
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.2)  # 添加学习率调度器
criterion = nn.CrossEntropyLoss()

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    model.to(device)
    running_loss = 0.0
    corrects = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        with torch.set_grad_enabled(False):
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == targets.data)
            running_loss += loss.item()
    
    return running_loss, corrects

# 将训练循环移到函数外部
    running_loss, corrects = train(model, device, dataloaders['train'], optimizer, epoch)  # 使用训练数据加载器
    print('Epoch: {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, running_loss / len(dataloaders['train'].dataset), corrects.double() / len(dataloaders['train'].dataset)))
    train(model, device, dataloaders['train'], optimizer, epoch)  # 使用训练数据加载器
    print('Epoch: {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, running_loss / len(dataloaders['train'].dataset), corrects.double() / len(dataloaders['train'].dataset)))


