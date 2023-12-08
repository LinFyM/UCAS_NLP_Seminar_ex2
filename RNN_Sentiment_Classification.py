import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import json
import argparse
from tqdm import tqdm
import torch.optim as optim
from collections import namedtuple
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, config) -> None:
        super(RNN, self).__init__()
        # 获取配置参数
        vocab_size = config.vocab_size
        embedding_dim = config.embedding_dim
        hidden_dim = config.hidden_dim
        output_dim = config.num_classes
        dropout_rate = config.dropout
        
        # 初始化文本embedding层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 初始化循环神经网络层(使用一层的RNN)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        
        # 初始化dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 初始化线性层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.transpose(0, 1)
        # print(f"x shape: {x.shape}")

        # 根据embedding层得到文本向量
        embedded = self.embedding(x)
        # print(f"embedded shape: {embedded.shape}")

        # 初始化RNN隐层向量(全0)
        h0 = torch.zeros(1, x.size(1), self.rnn.hidden_size, device=device)
        # print(f"h0 shape: {h0.shape}")

        # 输入至循环神经网络,得到最后的隐藏层表示 [batch_size, hidden_dim]
        output, hidden = self.rnn(embedded, h0)
        # print(f"hidden shape: {hidden.shape}")

        hidden = hidden[:, -1, :]

        # 应用dropout
        hidden = self.dropout(hidden)
        # hidden = self.dropout(hidden.squeeze(0))
        # print(f"hidden shape: {hidden.shape}")
        
        # 映射得到最终概率 [batch_size, output_dim]
        logits = self.fc(hidden)
        # print(f"logits shape: {logits.shape}")
        return logits

class SentimentDataset(Dataset):
    def __init__(self, data_path, vocab_path) -> None:
        self.vocab = json.load(open(vocab_path, 'r', encoding='utf-8'))
        self.data = self.load_data(data_path)
    
    def load_data(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = [json.loads(line) for line in f]
            random.shuffle(raw_data)
        data = []
        for item in raw_data:
            text = item['text']
            text_id = [self.vocab[t] if t in self.vocab.keys() else self.vocab['UNK'] for t in text]
            label = int(item['label'])
            data.append([text_id, label])
        return data
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
def train(model, config, train_dataset, eval_dataset):
    CE = nn.CrossEntropyLoss()
    # CE = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    model.train()
    global_step = 0
    best_acc = 0
    for epoch in range(config.num_epoch):
        progress_bar = tqdm(train_dataset, desc=f'Epoch {epoch}')
        for data in progress_bar:
            inputs, labels = data[0].to(device), data[1].to(device)
            logits = model(inputs)

            # print(logits.shape, labels.shape)

            loss = CE(logits, labels)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            global_step += 1
            if global_step % config.eval_interval == 0:
                progress_bar.set_postfix({'Loss': loss.item()})
        acc = evaluate(model, eval_dataset)
        print(f"Accuracy after epoch {epoch}: {acc}")
        if acc > best_acc:
            best_acc = acc
            save_dir = os.path.dirname(config.save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(model.state_dict(), config.save_path)
    return

def evaluate(model, eval_dataset):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 在评估模式下，我们不需要计算梯度
        for data in eval_dataset:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    model.train()  # 设置模型为训练模式
    return accuracy

def collate_fn(data):
    pad_idx = 8091
    texts = [d[0] for d in data]
    label = [d[1] for d in data]
    batch_size = len(texts)
    max_length = max([len(t) for t in texts])
    text_ids = torch.ones((batch_size, max_length)).long().fill_(pad_idx)
    label_ids = torch.tensor(label).long()
    for idx, text in enumerate(texts):
        text_ids[idx, :len(text)] = torch.tensor(text)
    return text_ids, label_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', default='save_model/best.pt')
    parser.add_argument('--train', default='./train.jsonl')
    parser.add_argument('--test', default='./test.jsonl')
    parser.add_argument('--val', default='./val.jsonl')
    parser.add_argument('--num_epoch', default=25, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--eval_interval', default=100, type=int)
    parser.add_argument('--vocab', default='./vocab.json')
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)

    arg = parser.parse_args()

    train_dataset = SentimentDataset(arg.train, arg.vocab)
    val_dataset = SentimentDataset(arg.val, arg.vocab)
    test_dataset = SentimentDataset(arg.test, arg.vocab)
    train_loader = DataLoader(train_dataset, batch_size=arg.batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=arg.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)

    # print('Total samples:', len(train_dataset))
    # print('Total batches:', len(train_loader))
    # for i, data in enumerate(train_loader):
    #     print('Batch', i, 'size:', data[0].size(0))

    chr_vocab = json.load(open(arg.vocab, 'r', encoding='utf-8'))

    config = {
        'dropout':arg.dropout,
        'num_classes':2,
        'vocab_size':len(chr_vocab),
        'embedding_dim':arg.hidden_dim,
        'hidden_dim':256,
    }

    config = namedtuple('config', config.keys())(**config)

    # 初始化模型
    model = RNN(config).to(device)

    # 训练模型
    train(model, arg, train_loader, val_loader)

    # 加载最优模型
    model.load_state_dict(torch.load(arg.save_path))

    # 评估模型
    accuracy = evaluate(model, test_loader)
    print('Test Accuracy: {:.2f}%'.format(accuracy * 100))    
    print('Learning Rate: {}'.format(arg.lr))
    print('Dropout Rate: {}'.format(arg.dropout))
