import torch.nn as nn
import torch
from typing import Tuple, Optional
from .sublayers import SelfAttentionLayer, FeedForwardLayer

class SelfAttentionEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = SelfAttentionLayer(d_model, num_heads, dropout)
        self.ffn = FeedForwardLayer(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, key_padding_mask=None):
        x, attn_weights = self.self_attn(x, key_padding_mask=key_padding_mask, attn_mask=None)
        x = self.ffn(x)
        return x, attn_weights
