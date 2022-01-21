from distutils.command.config import config
import torch
import torch.nn as nn
import numpy as np
import logging
    
import config 

from PositionalEncoding import Embedding, PositionalEncoding

logger = config.logging.getLogger('models')
fh= logging.FileHandler(config.log_file)
logger.addHandler(fh)

class Transformer(nn.Module):
    def __init__(self, nuc_vocab, max_len, attn_heads, embd_matrix ):
        super(Transformer,self ).__init__()
        self.config = config
        
        # attn = MultiHeadAttention(hidden, d_model)
        # ff = PositionalEncoding()
        self.position =  PositionalEncoding(config.dropout_size, config.d_model, max_len)
        self.embedding = Embedding(nuc_vocab, config.use_pretrained_embd, config.embedding_dims, embd_matrix, config.d_model)
        # self.pe_embd = nn.Sequential(embedding, position)
        
        
        
        
    def forward(self, x):
    
        x = self.embedding(x) #permute(1,0) if the shape of data is not bz x seq_length
        x = self.position(x)
        return 0
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        