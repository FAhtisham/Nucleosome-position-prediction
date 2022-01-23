import csv
import collections
from tqdm import tqdm
import numpy as np
import random

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import logging
from pycm import *
from collections import Counter 

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import sys

import config

# importing sys
import sys
  
# adding Folder_2 to the system path
sys.path.insert(0, '/home/abbasi/nucleosome_prediction')
from dataset import read_csv, convert_seqs_to_words, add_padding, ngram_counter, preprocess_data, create_loaders
from utils import create_embd_matrix, load_pretrained_embd
from Train import train, validate, test, reset_weights
from model import  CNNModel, CNNDynamicModel



# Logger
logger = config.logging.getLogger('main')
fh= logging.FileHandler(config.log_file)
logger.addHandler(fh)


# Read Dataset
data = read_csv(config.dataset_path)


# Create numpy arrays and encoding, decoding dictionaries
samples_np, labels_np, samples, encoding, decoding = preprocess_data(data, config.ngram)

# if pretrained embedding, then load the pretrained embedding vectors
if config.use_pretrained_embd:
    embd_dict= load_pretrained_embd(config.embd_file_path)
    embd_matrix= create_embd_matrix(encoding, embd_dict, config.embedding_dims) 

logger.info("Embedding and Dataset Loaded ...")

# Create dataloaders
train_dataloader, test_dataloader, train_ds, test_ds, full_dataset = create_loaders(samples_np, labels_np, samples, split_size=90)

# Declare Model
# if config.use_attention:
#     model = AttentionRNNModel(len(encoding), config.embedding_dims, max([len(i) for i in samples]), config.e_hidden_dims, embd_matrix, config.use_pretrained_embd, config.dropout_size, config.finetune)
# else:
#     model = RNNModel(len(encoding), config.embedding_dims, max([len(i) for i in samples]), config.e_hidden_dims, embd_matrix, config.use_pretrained_embd, config.dropout_size, config.finetune)
# def __init__(self,  filter_sizes = [7], num_filters = [100]):

# model=CNNModel(len(encoding), config.embedding_dims, embd_matrix, config.use_pretrained_embd)
model =   CNNDynamicModel( len(encoding), config.embedding_dims, embd_matrix)

if config.use_gpu:
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # 3e-4
loss_function = nn.BCELoss()



kfolds= 5
kfold= KFold(n_splits= kfolds, shuffle= True)
av_val_acc = 0.0
av_test_acc = 0.0

for fold, (train_idx, val_idx) in enumerate(kfold.split(train_ds),1):
    logger.info("____________ Train Fold No-{}___________".format(fold))

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    train_dataloader = DataLoader(full_dataset, batch_size = config.batch_size, sampler = train_subsampler)
    
    random.shuffle(val_idx)
    vald_idx = val_idx[:round(len(val_idx)/2)]
    test_idx = val_idx[round(len(val_idx)/2):]


    val_subsampler = torch.utils.data.SubsetRandomSampler(vald_idx)
    val_dataloader = DataLoader(full_dataset, batch_size = config.batch_size, sampler = val_subsampler)

    test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)
    test_dataloader = DataLoader(full_dataset, batch_size = config.batch_size, sampler = test_subsampler)

    
    for epoch in range(1, config.epochs):
        train(fold, train_dataloader, epoch, model, optimizer, loss_function)
    logger.info("\n\nValidate  \n\n")
    av_val_acc += validate(fold, val_dataloader, model, optimizer, loss_function)
    
    logger.info("\n\nTest  \n\n")
    av_test_acc += test(test_dataloader, model) 
    exit()
    model.apply(reset_weights)
    model.embedding.weight.data.copy_(torch.from_numpy(embd_matrix))

    # embd_dict= load_pretrained_embd(config.embd_file_path)
    # embd_matrix= create_embd_matrix(encoding, embd_dict, config.embedding_dims) 
    # model = AttentionRNNModel(len(encoding), config.embedding_dims, max([len(i) for i in samples]), config.e_hidden_dims, embd_matrix, config.use_pretrained_embd, config.dropout_size, config.finetune)
    # model=model.cuda()
logger.info("Avg  Val Accuracy : {}, \n Avg Test Accuracy {}".format(av_test_acc/kfolds, av_test_acc/kfolds))
    



