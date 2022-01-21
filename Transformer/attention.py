import torch
import torch.nn as nn
import torch.nn.functional as f



# Compute the attention scores and output from the attention and input 
def attention(query, key , value, mask = None, dropout = None):
    d_k = query.size(-1)
    scores = torch.matmul(q, key.transpose(-2,-1) ) # take transpose of k
    
    
    if mask is not None:
        scores = scores.masked_fill(mask ==0, 1e-9)
        
    p_attn = F.softmax(scores, dim = -1)
    
    if dropout is not None:
        p_attn = dropout(p_attn)
        
    
    return torch.matmul(p_attn, value), p_attn
        
    

    



class MultiHeadAttent(nn.Module):
    def __init__(self,d_model, attn_heads):
        super(MultiHeadAttent, self).__init__()
        
        assert d_mode % attn_heads == 0
        
        
        lin_list = [nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)]
        
        self.linear_layers = nn.Sequential(lin_list)
        
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.usqueeze(1)
        
        n_batches = query.size(0) 
        return 0
        
        
        
    