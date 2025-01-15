import torch

device = torch.device(
    'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
NUM_CHARS = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_EPOCHS = 15
BATCH_SIZE = 512
drop_out = 0.5
patience = 3
