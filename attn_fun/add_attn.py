import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    def __init__(self, query_size, key_size, hidden_size, drouput=0):
        super().__init__()
        self.W_q = nn.Linear(query_size, hidden_size, bias=False)
        self.W_k = nn.Linear(key_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(drouput)

    def forward(self, query, key, value, attn_mask=None):
        """
        Args:
            query: (N, n, d_q)
            key: (N, m, d_k)
            value: (N, m, d_v)
            attn_mask: broadcastable with (N, n, m)
        """
        query, key = self.W_q(query).unsqueeze(2), self.W_k(key).unsqueeze(1)
        scores = self.W_v(torch.tanh(query + key)).squeeze()  # (N, n, m)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, -1e9)  # 这样softmax后充分大的负数会变成0
        attn_weights = F.softmax(scores, dim=-1)  # (N, n, m)
        return self.dropout(attn_weights) @ value  # (N, n, d_v)
