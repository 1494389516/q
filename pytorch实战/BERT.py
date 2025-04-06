import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset

def load_data(batch_size, device):
    def preprocess_text(text):
        text = text.lower().strip()
        return ['<sos>'] + text.split() + ['<eos>']
    SRC = Field(tokenize = lambda x: x.split(), 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                pad_token='<pad>',
                lower = True)

    TRG = Field(tokenize = lambda x: x.split(), 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                pad_token='<pad>',
                lower = True)

    train_data, valid_data, test_data = TranslationDataset.splits(
    path = 'data/',
    exts = ('.src', '.trg'),
    fields = (SRC, TRG),
    filter_pred=lambda x: len(vars(x)['src']) <= 100 and len(vars(x)['trg']) <= 100  # 添加缺失的逗号
)
    SRC.build_vocab(train_data, min_freq = 2)
    TRG.build_vocab(train_data, min_freq = 2)
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = batch_size, 
        device = device,
         sort_key = lambda x: len(x.src),
        sort_within_batch = True
    )
    
    return train_iterator, valid_iterator, test_iterator, len(SRC.vocab), len(TRG.vocab), SRC, TRG
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class BERT(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, dropout):
        super(BERT, self).__init__()
        self.embedding = nn.Embedding(1000, hidden_size)
        self.position=nn.Embedding(1000,hidden_size)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout), num_layers)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        positions = torch.arange(0, x.size(1)).expand(x.size(0), x.size(1)).to(x.device)
        x = self.embedding(x) + self.position_embedding(positions)  # 添加位置编码
        x = self.dropout(x) 
        mask = (x == SRC.vocab.stoi['<pad>']).transpose(0, 1)  #生成padding 掩码
        x = x.permute(1, 0, 2)  # 调整维度顺序 (batch_size, seq_len) -> (seq_len, batch_size, dim)
        x = self.encoder(x)
        x = x.permute(1, 0, 2)  # 恢复维度顺序
        x = self.fc(x)
        return x

# 定义模型、优化器和损失函数
train_iterator, valid_iterator, test_iterator, src_vocab_size, trg_vocab_size, SRC, TRG = load_data(batch_size=32, device=device)
model = BERT(src_vocab_size, hidden_size=512, num_layers=6, num_heads=8, dropout=0.1).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
pad_idx = SRC.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx) 

# 计算训练时间
def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch[0].to(device)
            trg = batch[1].to(device)
            output = model(src)
            output = output.view(-1, output.shape[-1])
            trg = trg.view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# 计算训练时间
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# 训练模型
def train(model, iterator, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch[0].to(device)
        trg = batch[1].to(device)
        optimizer.zero_grad()
        output = model(src)
        output = output.view(-1, output.shape[-1])
        trg = trg.view(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# 管理训练模型的循环
def train_model(model, train_iterator, valid_iterator, optimizer, criterion, device, num_epochs):
    best_valid_loss = float('inf')
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = train(model, train_iterator, optimizer, criterion, device)
        valid_loss = evaluate(model, valid_iterator, criterion, device)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'bert-model.pt')

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')
train_model(model, train_iterator, valid_iterator, optimizer, criterion, device, num_epochs=10)

#加入评估模型和计算时间的代码,优化了该模型的训练时间
