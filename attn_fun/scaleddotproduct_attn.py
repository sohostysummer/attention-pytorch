import math
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        """
        Args:
            query: (N, n, d)
            key: (N, m, d)
            value: (N, m, d_v)
        """
        return F.softmax(query @ key.transpose(1, 2) / math.sqrt(query.size(2)), dim=-1) @ value
