import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """ 
    def __init__(self):
        '''
        Initialize the ScaledDotProductAttention class.
        '''
        # Initialize your softmax layer
        # What dimension should you pass to the softmax constructor?
        self.eps = 1e10 # DO NOT MODIFY
        self.softmax = NotImplementedError
        
    
    def forward(self, Q, K, V, mask=None):
        """
        :param Q: Query matrix of shape (N, ..., H, L, E) where L is target sequence length
        :param K: Key matrix of shape (N, ..., H, S, E) where S is source sequence length
        :param V: Value matrix of shape (N, ..., H, S, Ev) where Ev is value dimension
        :param mask: Boolean mask matrix of shape (N, ..., H, L, S) or broadcastable shape where 1/True indicates a position to ignore
        :return: Output matrix of shape (N, ..., H, L, Ev)
        """
        # TODO: Implement forward pass
        
        # Calculate attention scores
        scaled_dot_product = NotImplementedError
        
        # Apply mask before softmax if provided
        if mask is not None:
            scaled_dot_product = NotImplementedError

        # Compute attention scores: 
        # # Think about which dimension you should apply Softmax
        self.attention_scores = NotImplementedError

        # Calculate final output
        output = NotImplementedError

        # Return final output
        raise NotImplementedError
    
    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, ..., H, L, Ev)
        :return: Gradient of loss wrt input Q, K, V
        """
        # TODO: Implement backward pass

        # Calculate gradients for V
        d_V = NotImplementedError
        
        # Calculate gradients for attention scores
        d_attention_scores = NotImplementedError
        d_scaled_dot_product = NotImplementedError
        
        # Scale gradients by sqrt(d_k)
        d_scaled_dot_product = NotImplementedError
        
        # Calculate gradients for Q and K
        d_Q = NotImplementedError
        d_K = NotImplementedError
        
        # Return gradients for Q, K, V
        raise NotImplementedError

