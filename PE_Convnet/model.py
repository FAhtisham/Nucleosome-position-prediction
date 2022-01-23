import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import sys
from collections import OrderedDict
  
# adding Folder_2 to the system path
sys.path.insert(0, '/home/abbasi/nucleosome_prediction/Transformer')

from PositionalEncoding import Embedding, PositionalEncoding

class CNNModel(nn.Module):
    def __init__(self, nuc_vocab_size, embedding_size, embd_matrix, use_pretrained_embd, filter_sizes = [4,5,6], num_filters = [100,100,50]):
        super(CNNModel, self).__init__()
        self.nuc_vocab_size = nuc_vocab_size
        self.use_pretrained_embd = use_pretrained_embd
        # def __init__(self, nuc_vocab, embd_pretrained, embd_dims, embd_matrix, d_model ):

        # self.embedding = nn.Embedding(self.nuc_vocab_size, embedding_size)
        self.embedding=Embedding(self.nuc_vocab_size, config.use_pretrained_embd, config.embedding_dims, embd_matrix, config.d_model )

        self.position = PositionalEncoding(dropout=0, d_model=config.d_model, max_len=37)
        self.model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(100,100,5,1,2)),
            ('relu1', nn.ReLU())
            ]))
        self.dropout=nn.Dropout(0.2)
        self.Linear = nn.Linear(3700, 1)
        

    


    def forward(self, x):
        with torch.no_grad():
            embedded = self.embedding(x)
        embedded = self.position(embedded)
        posemd=embedded.permute(0,2,1)
        x=self.model(posemd)+posemd
        flat = torch.flatten(x, start_dim =1, end_dim=2)
        preds = self.Linear(self.dropout(flat))
        return F.softmax(preds.squeeze())



                                                                ########################################
                                                                ##### Have a look into this model#######
                                                                ########################################

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes,
                 output_dim, use_dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters,
                      kernel_size=(fs, embedding_dim)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(0.5 if use_dropout else 0.)

    def forward(self, x):
        x = x.permute(1, 0)
        embedded = self.embedding(x)
        embedded = embedded.unsqueeze(1)

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))

        return self.fc(cat)
    
    
class CNNDynamicModel(nn.Module):
  def __init__(self, nuc_vocab_size, embedding_size, embd_matrix, filter_sizes = [2,3,3,3,3], num_filters = [1000, 500, 200, 100, 100]):
    super(CNNDynamicModel, self).__init__()
    self.nuc_vocab_size = nuc_vocab_size


    self.embedding = Embedding(self.nuc_vocab_size, config.use_pretrained_embd, config.embedding_dims, embd_matrix, config.d_model )

    self.position = PositionalEncoding(dropout=0, d_model=config.d_model, max_len=37)
    

    self.conv1dList = nn.ModuleList([
                                     nn.Conv1d(in_channels = embedding_size, out_channels = num_filters[i],
                                     kernel_size = filter_sizes[i])
                                     for i in range(len(filter_sizes))
    ])
    # self.Linear = nn.Linear(np.sum(num_filters), 1)
    self.Linear1 = nn.Linear(13300, 1)
    self.rnn = nn.LSTM(1900, 100)
    self.Linear2 = nn.Linear(700, 1)
    self.dropout=nn.Dropout(0.2)

  def forward(self, x):
    # print("before per embedding : ", x.size())
    with torch.no_grad():
      embedded = self.embedding(x)
    embedded = self.position(embedded).permute(0,2,1)
    # print("before per embedding : ", embedded.size())
    
    # print("after per embedding : ", embedded.size())
    x_conv_list = [F.relu(conv1d(embedded)) for conv1d in self.conv1dList]
    # print("Conv size", x_conv_list[0].size(), "total conv", len(x_conv_list))
    x_pool_list = [F.max_pool1d(x_conv, kernel_size=5)
    for x_conv in x_conv_list]
    # for i in x_pool_list:
    #     print("pool", i.size())
    # print("pool size", x_pool_list[0].size(), "total pools", len(x_pool_list))
    # exit()
    x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)
    print("lin", x_fc.size())
    out,_ = self.rnn(x_fc.permute(0,2,1))
    print("lin size",torch.flatten(out, start_dim = 1, end_dim = 2).size())
    preds = self.Linear2(torch.flatten(out, start_dim = 1, end_dim = 2)) 
    # print("lin", x_fc.size(), torch.flatten(x_fc, start_dim = 1, end_dim = 2).size())
    # preds = self.Linear(self.dropout(torch.flatten(x_fc, start_dim=1, end_dim=2)))
    
    # print(preds.size())
    return F.softmax(preds.squeeze())
