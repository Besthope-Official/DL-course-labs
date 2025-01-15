import numpy as np
import torch.nn as nn
import torch


class PositionalEncoding(nn.Module):
    '''
        位置嵌入编码
    '''

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array([
            [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
            if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])
        self.pos_table = torch.FloatTensor(pos_table).cuda()

    def forward(self, enc_inputs):
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]
        return self.dropout(enc_inputs.cuda())


def get_attn_pad_mask(seq_q, seq_k):
    '''
        生成用于注意力机制的填充掩码
    '''
    batch_size, len_q = seq_q.size()
    _, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


class ScaledDotProductAttention(nn.Module):
    '''
        缩放点积注意力
    '''

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask, d_k):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)

        return context, attn


class MultiHeadAttention(nn.Module):
    '''
        多头注意力
    '''

    def __init__(self, d_model, d_k, n_heads, d_v):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.n_heads = n_heads
        self.d_v = d_v
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1,
                                   self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1,
                                   self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1,
                                   self.n_heads, self.d_v).transpose(1, 2)

        # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # context: [batch_size, n_heads, len_q, d_v]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask, self.d_k)
        # attn: [batch_size, n_heads, len_q, len_k]
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  self.n_heads * self.d_v)

        output = self.fc(context)

        return nn.LayerNorm(self.d_model).cuda()(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    '''
        前馈层
    '''

    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    # inputs: [batch_size, seq_len, d_model]
    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        # [batch_size, seq_len, d_model]
        return nn.LayerNorm(self.d_model).cuda()(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, n_heads, d_v, d_ff):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.n_heads = n_heads
        self.d_v = d_v
        self.d_ff = d_ff
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, n_heads, d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    # enc_inputs: [batch_size, src_len, d_model]
    def forward(self, enc_inputs, enc_self_attn_mask):
        # Attention block
        # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               # enc_outputs: [batch_size, src_len, d_model],
                                               # attn: [batch_size, n_heads, src_len, src_len]
                                               enc_self_attn_mask)
        # Add + Norm
        # enc_outputs: [batch_size, src_len, d_model]
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    '''
        Transformer 的编码器
    '''

    def __init__(self, d_model, src_vocab_size, n_layers, d_k, n_heads, d_v, d_ff):
        super(Encoder, self).__init__()
        # 转换词向量
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        # 加入位置信息
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, d_k, n_heads, d_v, d_ff) for _ in range(n_layers)])

    # enc_inputs: [batch_size, src_len]
    def forward(self, enc_inputs):
        # enc_outputs: [batch_size, src_len, d_model]
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs)

        # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class Transformer(nn.Module):
    '''
        Encoder-Only Transformer (BERT)
    '''

    def __init__(self, src_vocab_size, d_model, n_layers, n_heads, d_ff, num_class, d_k, d_v):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.n_heads = n_heads
        self.d_v = d_v
        self.d_ff = d_ff
        self.num_class = num_class
        self.num_layers = n_layers
        self.Encoder = Encoder(d_model, src_vocab_size,
                               n_layers, d_k, n_heads, d_v, d_ff).cuda()
        self.projection = nn.Linear(d_model, num_class, bias=False).cuda()

    def forward(self, enc_inputs, attention_mask):
        enc_outputs, enc_self_attns = self.Encoder(enc_inputs)
        cls_outputs = enc_outputs[:, 0, :]
        # 情感分析是分类问题
        output = self.projection(cls_outputs)
        return output
