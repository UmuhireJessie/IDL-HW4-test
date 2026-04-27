import torch.nn as nn
import torch 
from typing import Tuple, Optional

class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask=None, attn_mask=None):
        residual = x
        x_norm = self.norm(x)
        attn_out, attn_weights = self.mha(x_norm, x_norm, x_norm,
                                          key_padding_mask=key_padding_mask,
                                          attn_mask=attn_mask,
                                          need_weights=True,
                                          average_attn_weights=True)
        x = residual + self.dropout(attn_out)
        return x, attn_weights

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, y: torch.Tensor, key_padding_mask=None, attn_mask=None):
        residual = x
        x_norm = self.norm(x)
        attn_out, attn_weights = self.mha(x_norm, y, y,
                                          key_padding_mask=key_padding_mask,
                                          attn_mask=attn_mask,
                                          need_weights=True,
                                          average_attn_weights=True)
        x = residual + self.dropout(attn_out)
        return x, attn_weights

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = residual + self.dropout(self.ffn(self.norm(x)))
        return x
