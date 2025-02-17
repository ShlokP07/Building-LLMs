import torch.nn as nn
import torch 

class selfattention(nn.Module):
    def __init__(self, dim_in, dim_out, qkv_bias = False):
        super().__init__()
        
        ''''
        WE implement Q, V, K as a linear layer instaead of matrix as they are
        easier and better to train
        '''
        self.Q_W = nn.Linear(dim_in, dim_out, bias= qkv_bias)
        self.V_W = nn.Linear(dim_in, dim_out, bias= qkv_bias)
        self.K_W = nn.Linear(dim_in, dim_out, bias= qkv_bias)
        
    def forward(self, x):
        queries = self.Q_w(x)
        keys = self.K_W(x)
        values = self.V_W(x)
        
        attention_scores = torch.matmul(queries, keys.T)
        attention_weights  = torch.softmax(attention_scores/ keys[-1]**0.5, dim = -1)
        context_vector = torch.matmul(attention_weights, values)
        
        return context_vector
        
        
    
    