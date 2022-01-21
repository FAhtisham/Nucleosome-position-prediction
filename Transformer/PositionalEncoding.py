import math
import config
import logging


import torch
import torch.nn as nn


logger = config.logging.getLogger('PE')
fh= logging.FileHandler(config.log_file)
logger.addHandler(fh)


  


class Embedding(nn.Module):
    def __init__(self, nuc_vocab, embd_pretrained, embd_dims, embd_matrix, d_model ):
        super(Embedding, self).__init__()
        self.nuc_vocab = nuc_vocab+1
        self.embd_pretrained = embd_pretrained
        self.embd_matrix = embd_matrix
        self.d_model = d_model
        
        self.embd = nn.Embedding(self.nuc_vocab, embd_dims)
        
        if embd_pretrained:
            self.embd.weight.data.copy_(torch.from_numpy(self.embd_matrix))
            
        
    def forward(self, x):
        return self.embd(x)*math.sqrt(self.d_model)
    
    
    
  
        
        
        
class PositionalEncoding(nn.Module):
    def __init__(self, dropout, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        self.max_len = max_len
        print("d_model", d_model)
        pos_enc = torch.zeros(max_len, d_model)
        print("pe initial size", pos_enc.size())
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) #.view(-1,1)
        print("pos initial size", pos.size())
        div_term = torch.arange(0, d_model, 2).float() * -(math.log(10000)/d_model)
        print("div term  size", div_term.size())
        
        pos_enc[:,0::2] = torch.sin(torch.as_tensor(pos.numpy() * div_term.unsqueeze(0).numpy()))
        pos_enc[:,1::2] = torch.cos(torch.as_tensor(pos.numpy() * div_term.unsqueeze(0).numpy()))
        
        pos_enc = pos_enc.unsqueeze(0)
        print("final initial size", pos_enc.size())
        self.register_buffer('pos_enc', pos_enc)
        
    def forward(self, x):
        with torch.no_grad():
            print(x.size(1))
            print("smaller size", self.pos_enc[:,:x.size(1)].size())
            x = x+self.pos_enc[:,:x.size(1)]
        return x
        
        
        
        
        
        
            

