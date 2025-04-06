import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

train_path = r"D:\BaiduNetdiskDownload\第七章：新闻数据集文本分类实战\text\THUCNews\data\train.txt"
test_path = r"D:\BaiduNetdiskDownload\第七章：新闻数据集文本分类实战\text\THUCNews\data\test.txt"

# 定义 RNN 模型
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.rnn(embedded)
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        return self.fc(hidden)

# 自定义数据集
class TextDataset(Dataset):
    def __init__(self, file_path, vocab, tokenizer):
        self.data = []
        self.labels = []
        self.vocab = vocab
        self.tokenizer = tokenizer
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                label, text = line.strip().split('\t')
                self.data.append(text)
                self.labels.append(float(label == "pos"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]
        tokens = torch.tensor(self.vocab(self.tokenizer(text)), dtype=torch.int64)
        return tokens, label

def yield_tokens(data_iter):
    tokenizer = get_tokenizer("basic_english")
    for text in data_iter:
        yield tokenizer(text)

# 加载数据集
tokenizer = get_tokenizer("basic_english")
train_texts = [line.strip().split('\t')[1] for line in open(train_path, 'r', encoding='utf-8')]
vocab = build_vocab_from_iterator(yield_tokens(train_texts), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# 加载预训练词向量
embedding_path = r"D:\BaiduNetdiskDownload\第七章：新闻数据集文本分类实战\text\THUCNews\data\embedding_Tencent.npz"
embeddings = np.load(embedding_path)
embedding_matrix = torch.tensor(embeddings['embeddings'], dtype=torch.float32)

# 创建数据加载器
def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for _text, _label in batch:
        label_list.append(_label)
        text_list.append(_text)
        lengths.append(len(_text))
    label_list = torch.tensor(label_list, dtype=torch.float32)
    padded_text_list = pad_sequence(text_list, batch_first=True)
    return padded_text_list, label_list

train_dataset = TextDataset(train_path, vocab, tokenizer)
test_dataset = TextDataset(test_path, vocab, tokenizer)

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_batch),
    'test': DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_batch)
}

# 初始化模型参数
INPUT_DIM = len(vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

# 实例化模型
model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)

# 加载预训练词向量
pretrained_embeddings = vocab.get_vecs_by_index(torch.tensor(list(vocab.get_stoi().values())))
model.embedding.weight.data.copy_(pretrained_embeddings)

# 冻结预训练词向量
model.embedding.weight.requires_grad = False

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

# 训练模型
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        text, label = batch
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, label)
        acc = binary_accuracy(predictions, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# 计算准确率
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    return correct.sum() / len(correct)

# 训练循环
N_EPOCHS = 20
for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, dataloaders['train'], optimizer, criterion)
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc*100:.2f}%')