import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from config import device



class NameDataset(Dataset):
    def __init__(self, file_path, train=True, val=False):
        self.data = pd.read_csv(file_path, sep='\t')
        if train:
            self.data = self.data.sample(
                frac=0.8, replace=False, random_state=1, axis=0)
            self.data = self.data.reset_index(drop=True)
            self.len = self.data.shape[0]
        elif val:
            self.data = self.data.sample(
                frac=0.2, replace=False, random_state=1, axis=0)
            self.data = self.data.reset_index(drop=True)
            self.len = self.data.shape[0]
        self.phrase, self.sentiment = self.data['Phrase'], self.data['Sentiment']

    def __getitem__(self, index):
        # 根据数据索引获取样本
        return self.phrase[index], self.sentiment[index]

    def __len__(self):
        # 返回数据长度
        return self.len


def phrase2list(phrase):
    if isinstance(phrase, str):
        arr = [ord(c) for c in phrase]
    else:
        arr = [ord(c) for c in str(phrase)]
    return arr, len(arr)


def make_tensors(phrase, sentiment):
    sequences_and_lengths = [phrase2list(phrase) for phrase in phrase]
    phrase_sequences = [sl[0] for sl in sequences_and_lengths if sl[1] > 0]
    seq_lengths = torch.LongTensor(
        [sl[1] for sl in sequences_and_lengths if sl[1] > 0])
    sentiment = sentiment.long()

    seq_tensor = torch.zeros(len(phrase_sequences), seq_lengths.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(phrase_sequences, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
    seq_lengths, prem_idx = seq_lengths.sort(dim=0, descending=True)
    seq_tensor = seq_tensor[prem_idx]
    sentiment = sentiment[prem_idx]
    return seq_tensor.to(device), seq_lengths.to(device), sentiment.to(device)


def make_tensors1(phrase):
    sequences_and_lengths = [phrase2list(phrase) for phrase in phrase]
    phrase_sequences = [sl[0] for sl in sequences_and_lengths if sl[1] > 0]
    seq_lengths = torch.LongTensor(
        [sl[1] for sl in sequences_and_lengths if sl[1] > 0])

    seq_tensor = torch.zeros(len(phrase_sequences), seq_lengths.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(phrase_sequences, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
    seq_lengths, prem_idx = seq_lengths.sort(dim=0, descending=True)
    seq_tensor = seq_tensor[prem_idx]
    _, index = prem_idx.sort(descending=False)
    return seq_tensor.to(device), seq_lengths.to(device), index
