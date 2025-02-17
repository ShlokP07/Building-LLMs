import torch
from causal_attention import CausalAttention

torch.manual_seed(123)

# Define dimensions
d_in = 3   # Input dimension 
d_out = 2  # Output dimension 

# Create example input tensor
inputs = torch.tensor([
    [0.43, 0.15, 0.89],  # Your
    [0.55, 0.87, 0.66],  # journey
    [0.57, 0.85, 0.64],  # starts
    [0.22, 0.58, 0.33],  # with
    [0.77, 0.25, 0.10],  # one
    [0.05, 0.80, 0.55]   # step
])

#Two inputs with six tokens each; each token has embedding dimension 3
batch = torch.stack((inputs, inputs), dim=0)
print("Input shape:", batch.shape)  # [2, 6, 3] 


context_length = batch.shape[1]  # 6 tokens
ca = CausalAttention(d_in, d_out, context_length, dropout=0.0)
context_vecs = ca(batch)

# Print output shape
print("Output shape:", context_vecs.shape)  # [2, 6, 2] (batch_size, seq_length, d_out)

print("\nSample output (first batch, first token):")
print(context_vecs[0, 0])