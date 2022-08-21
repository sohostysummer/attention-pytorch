import torch.nn as nn
from attn_fun.dp_attn import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, bias=True, dropout=0):
        super().__init__()
        assert d_model % num_heads == 0
        self.hidden_size = d_model // num_heads
        self.num_heads = num_heads
        self.attn = ScaledDotProductAttention(dropout)
        self.W_q = nn.Linear(d_model, self.num_heads * self.hidden_size, bias=bias)
        self.W_k = nn.Linear(d_model, self.num_heads * self.hidden_size, bias=bias)
        self.W_v = nn.Linear(d_model, self.num_heads * self.hidden_size, bias=bias)
        self.W_o = nn.Linear(self.num_heads * self.hidden_size, d_model, bias=bias)

    def forward(self, query, key, value, attn_mask=None):
        """
        Args:
            query: (N, n, d_model)
            key: (N, m, d_model)
            value: (N, m, d_model)
            attn_mask: (N, n, m)
        """
        N, n, m = query.size(0), query.size(1), key.size(1)
        query, key, value = self.W_q(query), self.W_k(key), self.W_v(value)
        query = query.reshape(N, n, self.num_heads, self.hidden_size).permute(2, 0, 1, 3).reshape(-1, n, self.hidden_size)
        key = key.reshape(N, m, self.num_heads, self.hidden_size).permute(2, 0, 1, 3).reshape(-1, m, self.hidden_size)
        value = value.reshape(N, m, self.num_heads, self.hidden_size).permute(2, 0, 1, 3).reshape(-1, m, self.hidden_size)
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(self.num_heads, 1, 1)
        output = self.attn(query, key, value, attn_mask)
        output = output.reshape(self.num_heads, N, n, self.hidden_size).permute(1, 2, 0, 3).reshape(N, n, -1)
        return self.W_o(output)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, bias=True, dropout=0) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, bias=bias, dropout=dropout)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, X, attn_mask=None):
        """
        Args:
            X: (N, L, d_model)
            attn_mask: (N, n, m)
        """
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)
        return self.attn(Q, K, V, attn_mask)
