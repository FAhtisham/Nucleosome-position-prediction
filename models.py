import torch
import torch.nn as nn
import torch.nn.functional as F


from collections import OrderedDict


# RNN Model
class RNNModel(nn.Module):
  def __init__(self, nuc_pair_size, embedding_dims, seq_length, e_hidden_dims, embd_matrix, use_pretrained_embd, dropout_size, finetune):
    super(RNNModel, self).__init__()
    nuc_pair_size +=1
    self.seq_length= seq_length
    self.hidden_dims = e_hidden_dims
    self.use_pretrained_embd = use_pretrained_embd
    self.finetune = finetune
    self.embedding= nn.Embedding(nuc_pair_size, embedding_dims)
    
    if self.use_pretrained_embd:
        self.embedding.weight.data.copy_(torch.from_numpy(embd_matrix))
    self.rnn1= nn.LSTM(input_size= embedding_dims, hidden_size= e_hidden_dims, num_layers= 2, bidirectional= True)
    # self.a1= nn.ReLU(True)
    self.fc1= nn.Linear(in_features = e_hidden_dims*2, out_features = 1)
    # self.fc2= nn.Linear(in_features = 200, out_features = 1)
    self.dropout= nn.Dropout(dropout_size)

  def forward(self, x):
    if self.use_pretrained_embd and self.finetune:
      x= self.embedding(x).permute(1,0,2)
    elif self.use_pretrained_embd and not self.finetune:
      with torch.no_grad():
        x= self.embedding(x).permute(1,0,2)
    elif not self.use_pretrained_embd:
      x= self.embedding(x).permute(1,0,2)

    # Deals with the hidden state only 
    _,(hidden_states, _) = self.rnn1(x)
    hidden_states = torch.cat((hidden_states[-2,:,:], hidden_states[-1,:,:]), dim= 1)
    
    # # Deals with the output only 
    # output,(_, _) = self.rnn1(x)
    # # Combine forward and reverse output
    # output = torch.cat((output[x.size(0)-1,:,:self.hidden_dims], output[0,:,self.hidden_dims:]),1)

    lv= self.fc1(self.dropout(hidden_states)) 
    # lv= self.fc2(self.dropout(lv)) 
    lv= torch.sigmoid(lv)
    return lv.squeeze()



# 1D CNN Model (looks problematic) 
class CNNModel(nn.Module):
  def __init__(self, nuc_vocab_size, embedding_size, embd_matrix, use_pretrained_embd, filter_sizes = [7], num_filters = [100]):
    super(CNNModel, self).__init__()
    self.nuc_vocab_size = nuc_vocab_size+1
    self.use_pretrained_embd = use_pretrained_embd

    self.embedding = nn.Embedding(self.nuc_vocab_size, embedding_size)
    self.embedding.weight.data.copy_(torch.from_numpy(embd_matrix))

    self.model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv1d(100,100,8)),
          ('relu1', nn.ReLU()),
          ('flatten', nn.Flatten())
          ]))
    
    self.Linear = nn.Linear(4200, 1)
    self.dropout=nn.Dropout(0.2)

    


  def forward(self, x):
    # print("before per embedding : ", x.size())
    with torch.no_grad():
      embedded = self.embedding(x).permute(0,2,1)
    x=self.model(embedded)
    # x_fc = self.Linear(x)
    preds = self.Linear(self.dropout(x))
    return F.softmax(preds.squeeze())




# Attention RNN Model
class AttentionRNNModel(nn.Module):
  def __init__(self, nuc_pair_size, embedding_dims, seq_length, e_hidden_dims, embd_matrix, use_pretrained_embd, dropout_size, finetune):
      super(AttentionRNNModel, self).__init__()
      nuc_pair_size +=1
      self.seq_length= seq_length
      self.hidden_dims = e_hidden_dims
      self.use_pretrained_embd = use_pretrained_embd
      self.finetune=finetune
      self.embedding= nn.Embedding(nuc_pair_size, embedding_dims)
      if self.use_pretrained_embd:
          self.embedding.weight.data.copy_(torch.from_numpy(embd_matrix))
      self.rnn1= nn.LSTM(input_size= embedding_dims, hidden_size= e_hidden_dims, num_layers= 4, bidirectional= True)
      # self.a1= nn.ReLU(True)
      self.fc1= nn.Linear(in_features = e_hidden_dims*2, out_features = 1)
      # self.fc2= nn.Linear(in_features = 200, out_features = 1)
      self.dropout= nn.Dropout(dropout_size)

  def forward(self, x):
      if self.use_pretrained_embd and self.finetune:
        x= self.embedding(x).permute(1,0,2)
      elif self.use_pretrained_embd and not self.finetune:
        with torch.no_grad():
            x= self.embedding(x).permute(1,0,2)
      elif not self.use_pretrained_embd:
        x= self.embedding(x).permute(1,0,2)

      # Deals with the hidden state only 
      outputs,(hidden_states, _) = self.rnn1(x)
      # hidden_states = torch.cat((hidden_states[-2,:,:], hidden_states[-1,:,:]), dim= 1)
      
      attention_out = self.BhandauAttention(outputs, hidden_states)

      # # Deals with the output only 
      # # Combine forward and reverse output
      # output = torch.cat((output[x.size(0)-1,:,:self.hidden_dims], output[0,:,self.hidden_dims:]),1)

      lv= self.fc1(self.dropout(attention_out)) 
      # lv= self.fc2(self.dropout(lv)) 
      lv= torch.sigmoid(lv)
      return lv.squeeze()

  def BhandauAttention(self, output, hidden):
    # merge all hidden states
    merged_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim= 1)
    
    # Compute attention weights 
    weights = torch.bmm(output.permute(1,0,2), merged_hidden.unsqueeze(2))
    weights = F.softmax(weights.squeeze(2), 1).unsqueeze(2)
    
    # Compute and return the context vector
    new_hidden = torch.bmm(output.permute(1,2,0), weights.permute(0,1,2))
    return new_hidden.squeeze(2)







########################################
##### Have a look into this model#######
########################################



# class CNN(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes,
#                  output_dim, use_dropout):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.convs = nn.ModuleList([
#             nn.Conv2d(in_channels=1, out_channels=n_filters,
#                       kernel_size=(fs, embedding_dim)) for fs in filter_sizes
#         ])
#         self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
#         self.dropout = nn.Dropout(0.5 if use_dropout else 0.)

#     def forward(self, x):
#         x = x.permute(1, 0)
#         embedded = self.embedding(x)
#         embedded = embedded.unsqueeze(1)

#         conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
#         pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

#         cat = self.dropout(torch.cat(pooled, dim=1))

#         return self.fc(cat)