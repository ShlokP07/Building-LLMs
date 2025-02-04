import torch

def get_attention_scores(inputs):
    '''
    Calculate attention scores using dot product between inputs
    '''
    return torch.matmul(inputs, inputs.T)

def get_attention_weights(attention_scores):
    '''
    Convert scores to probabilities using softmax
    '''
    return torch.softmax(attention_scores, dim=-1)

def get_context_vector(inputs, attention_weights):
    '''
    Compute weighted sum of input vectors
    '''
    return torch.matmul(attention_weights, inputs)

def main():
    inputs = torch.tensor([
        [0.43, 0.15, 0.89],  # Your
        [0.55, 0.87, 0.66],  # journey
        [0.57, 0.85, 0.64],  # starts
        [0.22, 0.58, 0.33],  # with
        [0.77, 0.25, 0.10],  # one
        [0.05, 0.80, 0.55]   # step
    ])
    
    attention_scores = get_attention_scores(inputs)
    attention_weights = get_attention_weights(attention_scores)
    context_vector = get_context_vector(inputs, attention_weights)
    
    print(f'attention scores:\n{attention_scores}')
    print(f'attention weights:\n{attention_weights}')
    print(f'context vector:\n{context_vector}')

if __name__ == '__main__':
    main()