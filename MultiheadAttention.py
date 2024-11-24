import torch.nn as nn
import torch.nn.functional as F
import torch

def dot_product_attention(queries, keys, values):
    scores = torch.bmm(queries, keys.transpose(1, 2)) 
    attention_weights = F.softmax(scores, dim=-1)       
    output = torch.bmm(attention_weights, values)     
    return output, attention_weights

def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class MultiheadAttention(nn.Module):
    def __init__(self, key_size, value_size, query_size, num_heads, num_hidden, dropout, bias=False, **kwargs):
        super(MultiheadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = dot_product_attention
        self.W_q = nn.Linear(query_size, num_hidden, bias=bias)
        self.W_k = nn.Linear(key_size, num_hidden, bias=bias)
        self.W_v = nn.Linear(value_size, num_hidden, bias=bias)
        self.W_o = nn.Linear(num_hidden, query_size, bias=bias)

    def forward(self, queries, keys, values, Valid_Length):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
         
        if Valid_Length is not None:
            Valid_Length = torch.repeat_interleave(Valid_Length, self.num_heads, dim=0)

        output, attention_weights = self.attention(queries, keys, values)

        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)