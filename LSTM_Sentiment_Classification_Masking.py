import torch
import matplotlib.pyplot as plt
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

class LSTM(nn.Module):
    def __init__(self, config) -> None:
        super(LSTM, self).__init__()
        # 获取配置参数
        vocab_size = config.vocab_size
        embedding_dim = config.embedding_dim
        hidden_dim = config.hidden_dim
        output_dim = config.num_classes
        dropout_rate = config.dropout
        
        # 初始化文本embedding层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 初始化循环神经网络层(使用一层的LSTM)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        
        # 初始化dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 初始化线性层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths):
        x = x.transpose(0, 1)

        # 根据embedding层得到文本向量
        embedded = self.embedding(x)

        # 使用pack_padded_sequence 来打包输入序列
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths)

        # 输入至循环神经网络,得到最后的隐藏层表示
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # dropout
        hidden = self.dropout(hidden.squeeze(0))

        # 映射得到最终概率
        logits = self.fc(hidden)

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

accuracy_results = []

def train(model, config, train_dataset, eval_dataset):
    CE = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    model.train()
    global_step = 0
    best_acc = 0
    for epoch in range(config.num_epoch):
        progress_bar = tqdm(train_dataset, desc=f'Epoch {epoch}')
        for data in progress_bar:
            inputs, labels, lengths = data[0].to(device), data[1].to(device), data[2]
            logits = model(inputs, lengths)
            loss = CE(logits, labels)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            global_step += 1
            if global_step % config.eval_interval == 0:
                progress_bar.set_postfix({'Loss': loss.item()})
        acc = evaluate(model, eval_dataset)
        accuracy_results.append(acc)
        print(f"Accuracy after epoch {epoch}: {acc}")
        if acc > best_acc:
            best_acc = acc
            save_dir = os.path.dirname(config.save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(model.state_dict(), config.save_path)
    
    plt.plot(range(config.num_epoch), accuracy_results)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.show()
    
    return

def evaluate(model, eval_dataset):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad(): 
        for data in eval_dataset:
            inputs, labels, lengths = data[0].to(device), data[1].to(device), data[2]
            outputs = model(inputs, lengths)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    model.train()
    return accuracy

def collate_fn(data):
    pad_idx = 8019
    data.sort(key=lambda x: len(x[0]), reverse=True)
    texts = [d[0] for d in data]
    label = [d[1] for d in data]
    lengths = [len(t) for t in texts]
    batch_size = len(texts)
    max_length = max(lengths)
    text_ids = torch.ones((batch_size, max_length)).long().fill_(pad_idx)
    label_ids = torch.tensor(label).long()
    for idx, text in enumerate(texts):
        text_ids[idx, :len(text)] = torch.tensor(text)
    return text_ids, label_ids, lengths

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', default='save_model/best.pt')
    parser.add_argument('--train', default='./train.jsonl')
    parser.add_argument('--test', default='./test.jsonl')
    parser.add_argument('--val', default='./val.jsonl')
    parser.add_argument('--num_epoch', default=30, type=int)
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
    model = LSTM(config).to(device)

    # 训练模型
    train(model, arg, train_loader, val_loader)

    # 加载最优模型
    model.load_state_dict(torch.load(arg.save_path))

    # 评估模型
    accuracy = evaluate(model, test_loader)
    print('Test Accuracy: {:.2f}%'.format(accuracy * 100))    
    print('Learning Rate: {}'.format(arg.lr))
    print('Dropout Rate: {}'.format(arg.dropout))
    print('Batch Size: {}'.format(arg.batch_size))
