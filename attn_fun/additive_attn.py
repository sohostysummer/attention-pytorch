import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    def __init__(self, query_size, key_size, hidden_size):
        super().__init__()
        self.W_q = nn.Linear(query_size, hidden_size, bias=False)
        self.W_k = nn.Linear(key_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query, key, value):
        """
        Args:
            query: (N, n, d_q)
            key: (N, m, d_k)
            value: (N, m, d_v)
        """
        query, key = self.W_q(query).unsqueeze(2), self.W_k(key).unsqueeze(1)
        attn_weights = F.softmax(self.W_v(torch.tanh(query + key)).squeeze(), dim=-1)  # (N, n, m)
        return attn_weights @ value  # (N, n, d_v)
