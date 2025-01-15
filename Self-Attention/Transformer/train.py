import torch.nn as nn
import torch.optim as optim
from dataset import *
import csv
from torch.utils.data import Dataset, DataLoader
from transformer import Transformer
from transformers import BertTokenizer
from config import *


def load_model():
    model = torch.load('model.ptl')
    model.to(device)
    return model


if __name__ == "__main__":

    phrases = []
    labels = []
    with open(r'data/train.tsv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)
        for row in reader:
            if len(row) >= 3 and isinstance(row[2], str):
                phrases.append(row[2])
                labels.append(int(row[3]))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = EmotionDataset(phrases, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    
    model = Transformer(src_vocab_size, d_model, n_layers,
                        n_heads, d_ff, num_class, d_k, d_v).cuda()
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        loss_list = []
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if batch_idx % 1000 == 0:
                total_loss += loss.item()
            average_loss = total_loss / (len(dataloader) // 1000)
        print(f"Epoch {epoch + 1}, Loss: {average_loss:.4f}")
        loss_list.append(average_loss)
        if average_loss <= min(loss_list):
            torch.save(model, 'model.ptl')
            print('Save Model!')
