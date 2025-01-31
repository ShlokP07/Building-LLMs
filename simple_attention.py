import torch

def calculate_attention_scores(query, inputs):
    '''
    init an attention tensor
    calculate the dot product between query and each word
    '''
    attention_scores = torch.empty(inputs.shape[0])
    for idx, x in enumerate(inputs):
        attention_scores[idx] = torch.dot(x, query)
    
    return attention_scores

def calculate_attention_weights(attention_scores):
    '''
    convert raw scores that sum up to 1
    '''
    return torch.softmax(attention_scores, dim=0) # remember to calculate softmax along each row

def main():
    inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your    
    [0.55, 0.87, 0.66], # journey  
    [0.57, 0.85, 0.64], # starts  
    [0.22, 0.58, 0.33], # with    
    [0.77, 0.25, 0.10], # one     
    [0.05, 0.80, 0.55]] # step    
    )

    query = inputs[1] 
    
    attention_scores = calculate_attention_scores(query, inputs)
    attention_weights = calculate_attention_weights(attention_scores)
    print(f'attention scores: {attention_scores}')
    print(f'attention weights: {attention_weights}')      

    
if __name__ == '__main__':
    main()
    