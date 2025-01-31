import torch

def get_attention_scores(inputs):
    '''
    init an attention tensor
    calculate the dot product between query and each word
    '''
    attention_scores = torch.empty(inputs.shape[0], inputs.shape[0])
    attention_scores = torch.matmul(inputs, inputs.T)
    
    # for i, x_i in enumerate(inputs):
    #     for j, x_j in enumerate(inputs):
    #         attention_scores[i, j] = torch.dot(x_i, x_j)
    
    return attention_scores

def get_attention_weights(attention_scores):
    '''
    convert raw scores that sum up to 1
    '''
    return torch.softmax(attention_scores, dim=-1) # we tell pytorch to apply softmax on the last dim, robust, works well at higher dim

def get_context_vector(inputs, attention_weights):
    '''
    multiply each index of each imput word with its attention weight and then add
    '''
    return torch.matmul(attention_weights, inputs) # remember when matmul you have to match inner dimensions thats y this order

def main():
    inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your    
    [0.55, 0.87, 0.66], # journey  
    [0.57, 0.85, 0.64], # starts  
    [0.22, 0.58, 0.33], # with    
    [0.77, 0.25, 0.10], # one     
    [0.05, 0.80, 0.55]] # step    
    )

    
    attention_scores = get_attention_scores(inputs)
    attention_weights = get_attention_weights(attention_scores)
    convext_vector = get_context_vector(attention_weights)
    print(f'attention scores: {attention_scores}')
    print(f'attention weights: {attention_weights}')      
    print(f'context vector: {convext_vector}')     

    
if __name__ == '__main__':
    main()
    