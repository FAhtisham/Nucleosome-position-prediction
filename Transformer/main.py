import csv
import collections
from tqdm import tqdm
import numpy as np
import random

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from pycm import *
from collections import Counter 

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import sys
import config
from tqdm import tqdm

from models import Transformer
import logging


from dataset import read_csv, convert_seqs_to_words, add_padding, ngram_counter, preprocess_data, create_loaders
from utils import create_embd_matrix, load_pretrained_embd
from Train import train, validate, test

logger = config.logging.getLogger('main')
fh= logging.FileHandler(config.log_file)
logger.addHandler(fh)

# Read Dataset
data = read_csv(config.dataset_path)




# Create numpy arrays and encoding, decoding dictionaries
samples_np, labels_np, samples, encoding, decoding = preprocess_data(data[1:], config.ngram)

# Find max len
max_len = 0.0

for i in range(len(samples_np)):
    if (len(samples_np[i]))>max_len:
        max_len = len(samples_np[i])
logger.info("Max len sequence : {}".format(max_len))

# if pretrained embedding, then load the pretrained embedding vectors
if config.use_pretrained_embd:
    embd_dict= load_pretrained_embd(config.embd_file_path)
    embd_matrix= create_embd_matrix(encoding, embd_dict, config.embedding_dims) 
else:
    embd_matrix = None

logger.info("Embeddings loadded and dataset as well")


t_model = Transformer(len(encoding), max_len, config.attn_heads, embd_matrix)

if torch.cuda.is_available():
    t_model = t_model.cuda()




# Create dataloaders
train_dataloader, test_dataloader, train_ds, test_ds, train_DataLoader_FD, full_dataset = create_loaders(samples_np, labels_np, samples, split_size=90)

optimizer = torch.optim.Adam(t_model.parameters(), lr=0.001)
loss_function = nn.BCELoss()


kfolds= 5
kfold= KFold(n_splits= kfolds, shuffle= True)


# K fold val loop
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
        train(fold, train_dataloader, epoch, t_model, optimizer, loss_function)
        exit()
        
    logger.ifno("\n\nValidate  \n\n")
    validate(fold, val_dataloader, classifier, optimizer, loss_function)
    
    logger.info("\n\nTest  \n\n")
    test(test_dataloader, classifier) 

    classifier.apply(reset_weights)


