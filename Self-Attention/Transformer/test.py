import torch
import csv
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from train import load_model
from config import batch_size, device
from dataset import EmotionDatasetest

if __name__ == "__main__":
    phrases = []
    with open(r'data/test.tsv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)
        for row in reader:
            if len(row) >= 2 and isinstance(row[1], str):
                phrases.append(row[1])
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = EmotionDatasetest(phrases, tokenizer)
    dataloader = DataLoader(dataset, batch_size, shuffle=False)

    # use pre-trained
    model = load_model()

    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, dim=1)
            predictions.extend(predicted.cpu().numpy())

    with open('data/predictions.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['PhraseId', 'Sentiment'])
        for i, pred in enumerate(predictions):
            writer.writerow([i+1, pred])
