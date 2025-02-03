import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, dim_in, dim_out, context_length, dropout, qkv_bias = False):
        super().__init__()
        
        self.dim_in = dim_in
        
        self.Q_W = nn.Linear(dim_in, dim_out, bias= qkv_bias)
        self.K_W = nn.Linear(dim_in, dim_out, bias= qkv_bias)
        self.V_W = nn.Linear(dim_in, dim_out, bias= qkv_bias)
        
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
        
    def forward(self, x):
        pass
        
    
        