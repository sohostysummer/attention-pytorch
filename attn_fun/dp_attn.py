import math
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        """
        Args:
            query: (N, n, d_k)
            key: (N, m, d_k)
            value: (N, m, d_v)
            attn_mask: (N, n, m)
        """
        assert query.size(2) == key.size(2)
        scores = query @ key.transpose(1, 2) / math.sqrt(query.size(2))
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        return self.dropout(attn_weights) @ value
