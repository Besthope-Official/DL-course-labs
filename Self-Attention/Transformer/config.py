import torch

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# 嵌入维度
d_model = 512
# FFN 的维度
d_ff = 2048
d_k = d_v = 64
# encoder 的层数
n_layers = 6
# 多头注意力的头数
n_heads = 8

src_vocab_size = 50000

# 训练超参
num_epochs = 10
num_class = 5       # 5 种分类
batch_size = 128
