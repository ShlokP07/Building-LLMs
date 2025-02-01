import torch.nn as nn

class selfattention(nn.Moudule):
    def __init__(self, dim_in, dim_out, qkv_bias = False):
        super().__init__()
        
        ''''
        WE implement Q, V, K as a linear layer instaead of matrix as they are
        easier and better to train
        '''
        q_W = nn.Linear(dim_in, dim_out, bias= qkv_bias)
        V_W = nn.Linear(dim_in, dim_out, bias= qkv_bias)
        K_W = nn.Linear(dim_in, dim_out, bias= qkv_bias)
        
    def forward(self, x):
        pass
    
    