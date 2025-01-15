import torch
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    def __init__(self, enc_inputs, labels, tokenizer, max_len=128):
        super(EmotionDataset, self).__init__()
        self.enc_inputs = enc_inputs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.enc_inputs)

    def __getitem__(self, idx):
        enc_input = self.enc_inputs[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            enc_input,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class EmotionDatasetest():
    def __init__(self, phrases, tokenizer):
        self.phrases = phrases
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, idx):
        phrase = self.phrases[idx]
        inputs = self.tokenizer(phrase, return_tensors='pt',
                                max_length=512, padding='max_length', truncation=True)
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
