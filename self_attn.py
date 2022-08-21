import torch.nn as nn
from attn_fun.dp_attn import ScaledDotProductAttention


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, key_size, value_size, dropout=0):
        super().__init__()
        self.attn = ScaledDotProductAttention(dropout)
        self.W_q = nn.Linear(embed_dim, key_size, bias=False)  # 因为query和key的长度相同，否则无法使用点积注意力
        self.W_k = nn.Linear(embed_dim, key_size, bias=False)
        self.W_v = nn.Linear(embed_dim, value_size, bias=False)

    def forward(self, X, attn_mask=None):
        """
        Args:
            X: input sequence, shape: (N, L, embed_dim)
            attn_mask: (N, L, L)
        """
        Q = self.W_q(X)  # (N, L, key_size)
        K = self.W_k(X)  # (N, L, key_size)
        V = self.W_v(X)  # (N, L, value_size)
        return self.attn(Q, K, V, attn_mask)  # (N, L, value_size)
