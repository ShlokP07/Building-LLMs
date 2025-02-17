import torch
import torch.nn as nn

class MultiHeadAttention(nn.Moudle):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        
        assert((d_out % num_heads==0), "d_out must be divisible by num_heads")
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads   
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)   
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)        
        queries = self.W_query(x)   
        values = self.W_value(x) 

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)      
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)  
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)     
                                                                           
        keys = keys.transpose(1, 2)         
        queries = queries.transpose(1, 2)   
        values = values.transpose(1, 2)  

        attn_scores = queries @ keys.transpose(2, 3)  
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]   
        attn_scores.masked_fill_(mask_bool, -torch.inf)    
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context_vec = (attn_weights @ values).transpose(1, 2)  
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)   
        
        return context_vec

"""
When we apply dropout after softmax, we're essentially randomly zeroing out some of the attention weights after they've been normalized to sum to 1. This has some key advantages:
First, in terms of the mathematics, applying dropout after softmax maintains better statistical properties. The softmax operation ensures all weights sum to 1, creating a proper probability distribution. When we then apply dropout, we're randomly dropping some of these probabilities, but the remaining weights still maintain meaningful relative proportions to each other. If we were to apply dropout before softmax, we might end up with many negative infinity values (from zeroed inputs), which could lead to unstable gradients and training difficulties.
Second, from a modeling perspective, dropping attention weights after softmax effectively forces the model to not rely too heavily on any single relationship between tokens. Think of it this way: if a token was paying very strong attention to one specific other token, dropout might occasionally remove that connection, forcing the model to also learn to use information from other tokens as backup. This creates a more robust and generalized attention mechanism.
"""