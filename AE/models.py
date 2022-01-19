import torch
import torch.nn as nn
import torch.nn.functional as f

class EmbeddingBlock(nn.Module):
    def __init__(self, nuc_vocab_size, embd_dims, embd_matrix):
        super(EmbeddingBlock, self).__init__()
        
        self.nuc_vocab_size = nuc_vocab_size+1
        self.embd = nn.Embedding(self.nuc_vocab_size, embd_dims)
        self.embd.weight.data.copy_(torch.from_numpy(embd_matrix))

        
    def forward(self, x):
        # with torch.no_grad():
#             print("before embd: ", x.size())
        embd = self.embd(x).permute(1,0,2)
#             print("after embd: ", embd.size())
        return embd
        
        
        
class EncoderBlock(nn.Module):
    def __init__(self, embd_dims, hidden_dims, nlayers):
        super(EncoderBlock, self).__init__()
        self.hidden = hidden_dims
        self.nlayers = nlayers
        
        self.rnn = nn.LSTM(input_size=embd_dims, hidden_size = hidden_dims, num_layers = nlayers, bidirectional = False)
#         self.btn = nn.Linear(in_features = hidden_dims * nlayers, out_features = 100 )
        
    def forward(self, x):
        h0 = torch.zeros(self.nlayers, x.size(1), self.hidden).cuda()
        c0 = torch.zeros(self.nlayers, x.size(1), self.hidden).cuda()
        outputs, (hidden, cell_state) = self.rnn(x, (h0, c0))
        
        return outputs, hidden
    
    
class DecoderBlock(nn.Module):
    def __init__(self,  nuc_pair_size, embd_dims, hidden_dims, nlayers, out_dims):
        super(DecoderBlock, self).__init__()
        self.hidden = hidden_dims
        self.nlayers = nlayers
        
        self.rnn = nn.LSTM(input_size=out_dims, hidden_size = hidden_dims, num_layers = nlayers, bidirectional = False)
        self.lin = nn.Linear(in_features = hidden_dims * nlayers, out_features = nuc_pair_size )
        
    def forward(self, x):
        outputs, (hidden, cell_state) = self.rnn(x)
        outputs = self.lin(outputs)
        return outputs

    
class AE(nn.Module):
    def __init__(self, nuc_vocab_size, embd_dims, hidden_dims, nlayers, out_dims):
        super(AE, self).__init__()       
        self.encoder_block = EncoderBlock( embd_dims, hidden_dims, nlayers)
        self.decoder_block = DecoderBlock(nuc_vocab_size, embd_dims, hidden_dims, nlayers, out_dims)
  
    def forward(self, x):
        encoded, _ = self.encoder_block(x)
        decoded = self.decoder_block(encoded)
        return decoded


class Classifier(nn.Module):
    def __init__(self, hidden_dims, models_list, embedding):
        super(Classifier, self).__init__()
        self.embedding = embedding
        self.encoder1= models_list[0].encoder_block
        self.encoder2= models_list[1].encoder_block
        self.encoder3= models_list[2].encoder_block
        self.lin = nn.Linear(in_features = hidden_dims, out_features = 1)
        
        
    def forward(self, x):
        with torch.no_grad():
            embd = self.embedding(x)
        x_out, _ = self.encoder1(embd)
        x_out, _ = self.encoder2(x_out)
        x_out, _ = self.encoder3(x_out)
        logits = self.lin(x_out[-1,:,:])
        return nn.functional.sigmoid(logits.squeeze()) 
        