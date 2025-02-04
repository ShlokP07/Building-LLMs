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
        
        b, num_tokens, d_in = x.shape
        
        queries = self.Q_W(x)
        keys = self.K_W(x)
        values = self.V_W(x)
        
        attention_scores = torch.matmul(queries, keys.transpose(1, 2))
        attention_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context_vector = torch.matmul(attention_weights, values)
        
        return context_vector
            
        
    
        