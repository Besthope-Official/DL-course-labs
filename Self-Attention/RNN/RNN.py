import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from config import device


class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_direction = 2 if bidirectional else 1

        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size,
                                n_layers, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = torch.nn.Linear(hidden_size*self.n_direction, output_size)

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers*self.n_direction,
                             batch_size, self.hidden_size)
        return hidden.to(device)

    def forward(self, input, seq_lengths):
        input = input.t()
        batch_size = input.size(1)
        hidden = self._init_hidden(batch_size)
        embedding = self.embedding(input)

        gru_input = pack_padded_sequence(embedding, seq_lengths.cpu())
        output, hidden = self.gru(gru_input, hidden)
        if self.n_direction == 2:
            hidden_cat = torch.cat((hidden[-1], hidden[-2]), dim=1)
        else:
            hidden_cat = hidden[-1]
        fc_output = self.fc(hidden_cat)
        return fc_output
