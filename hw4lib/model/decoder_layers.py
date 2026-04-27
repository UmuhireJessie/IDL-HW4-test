import torch.nn as nn
import torch
from typing import Tuple, Optional
from .sublayers import SelfAttentionLayer, CrossAttentionLayer, FeedForwardLayer

class SelfAttentionDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = SelfAttentionLayer(d_model, num_heads, dropout)
        self.ffn = FeedForwardLayer(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, key_padding_mask=None, attn_mask=None):
        x, attn_weights = self.self_attn(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = self.ffn(x)
        return x, attn_weights

class CrossAttentionDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = SelfAttentionLayer(d_model, num_heads, dropout)
        self.cross_attn = CrossAttentionLayer(d_model, num_heads, dropout)
        self.ffn = FeedForwardLayer(d_model, d_ff, dropout)

    def forward(self, x, enc_output, dec_key_padding_mask=None, enc_key_padding_mask=None, attn_mask=None):
        x, self_attn_weights = self.self_attn(x, key_padding_mask=dec_key_padding_mask, attn_mask=attn_mask)
        x, cross_attn_weights = self.cross_attn(x, enc_output, key_padding_mask=enc_key_padding_mask)
        x = self.ffn(x)
        return x, self_attn_weights, cross_attn_weights
