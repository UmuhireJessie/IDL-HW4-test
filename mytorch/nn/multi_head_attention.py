from .linear import Linear
from .scaled_dot_product_attention import ScaledDotProductAttention
import numpy as np

class MultiHeadAttention:
    """
    Multi Head Attention
    """ 
    def __init__(self, embed_dim, num_heads):
        """
        :param embed_dim: Embedding dimension
        :param num_heads: Number of attention heads
        """
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        # Initialize parameters and layers
        # DO NOT MODIFY
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Initialize your scaled dot product attention layer
        self.attention = NotImplementedError
        
        # Initialize your linear layer
        #  embed_dim -> embed_dim
        self.q_proj   = NotImplementedError
        self.k_proj   = NotImplementedError
        self.v_proj   = NotImplementedError
        self.out_proj = NotImplementedError

    def init_weights(self, Wq, bq, Wk, bk, Wv, bv, Wo, bo):
        """
        Initialize the weights and biases with the given values.
        """
        # Initialize your linear layers (DO NOT MODIFY)
        self.q_proj.init_weights(Wq, bq)
        self.k_proj.init_weights(Wk, bk)
        self.v_proj.init_weights(Wv, bv)
        self.out_proj.init_weights(Wo, bo)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        :param query: (N, L, E)
        :param key: (N, S, E)
        :param value: (N, S, E)
        :param key_padding_mask: (N, S) where 1/True indicates positions to ignore
        :param attn_mask: (L, S) where 1/True indicates positions to ignore
        :return: (N, L, E)
        """
        
        # TODO: Implement forward pass

        self.N = query.shape[0]
        self.L = query.shape[1]
        self.S = key.shape[1]
        self.E = query.shape[2]
        
        # Project inputs
        q = NotImplementedError
        k = NotImplementedError
        v = NotImplementedError

        # Reshape for multiple heads
        q = NotImplementedError
        k = NotImplementedError
        v = NotImplementedError

        # Combine padding and causal masks
        mask = NotImplementedError

        # Apply attention
        attn_outputs = NotImplementedError

        # Merge heads
        attn_output = NotImplementedError

        # Final projection
        output = NotImplementedError

        raise NotImplementedError

    def backward(self, d_output):
        """
        Backward pass for multi-head attention.
        """

        # Backpropagate through output projection
        d_attn_output = NotImplementedError

        # Undo head splitting
        d_attn_outputs = NotImplementedError

        # Backpropagate through attention
        d_q, d_k, d_v = NotImplementedError

        # Merge head gradients
        d_q = NotImplementedError
        d_k = NotImplementedError
        d_v = NotImplementedError

        # Backpropagate through input projections
        d_q = NotImplementedError
        d_k = NotImplementedError
        d_v = NotImplementedError

        raise NotImplementedError

    def _merge_masks(self, key_padding_mask, attn_mask):
        """
        Merge two mask types into a single mask.
        """
        # Expand masks for broadcasting
        key_mask = NotImplementedError
        attention_mask = NotImplementedError
        
        # Combine masks
        combined_mask = NotImplementedError
        
        raise NotImplementedError

    def _split_heads(self, x):
        """
        Reshape tensor for multi-head attention.
        """
        # Reshape and transpose for heads
        x = NotImplementedError
        x = NotImplementedError
        
        raise NotImplementedError

    def _concat_heads(self, x):
        """
        Concatenate the last dimension into (num_heads, d_k).
        Transpose to move num_heads dimension to the back.
        :param x: (N, num_heads, L, embed_dim // num_heads)
        :return: (N, L, embed_dim)
        """
        # Transpose and reshape
        x = NotImplementedError
        x = NotImplementedError
        
        raise NotImplementedError
