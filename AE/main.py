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
from models import AE, Classifier, EmbeddingBlock
from dataset import read_csv, convert_seqs_to_words, add_padding, ngram_counter, preprocess_data, create_loaders
from utils import create_embd_matrix, load_pretrained_embd
from train import train, validate, test, reset_weights, TrainClass



# Read Dataset
data = read_csv(config.dataset_path)


# Create numpy arrays and encoding, decoding dictionaries
samples_np, labels_np, samples, encoding, decoding = preprocess_data(data[1:], config.ngram)

# if pretrained embedding, then load the pretrained embedding vectors
if config.use_pretrained_embd:
    embd_dict= load_pretrained_embd(config.embd_file_path)
    embd_matrix= create_embd_matrix(encoding, embd_dict, config.embedding_dims) 

print("Embeddings loadded and dataset as well",  file=open("log.txt", "a"))


# Create dataloaders
train_dataloader, test_dataloader, train_ds, test_ds, train_DataLoader_FD, full_dataset = create_loaders(samples_np, labels_np, samples, split_size=90)


#Create embeddings
input_embd = EmbeddingBlock(len(encoding), config.embedding_dims, embd_matrix)

# Declare AE Models (3 in this case)
ae_list = [AE(len(encoding), config.embedding_dims, config.hidden_dims, config.nlayers, config.out_dims) for i in range(3)]

# Define a loss function for them
loss_fn = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    ae_list = [ae_list[i].cuda() for i in range(3)]
    input_embd= input_embd.cuda()

if config.use_pretrained_aes:
    for i, m in enumerate(ae_list):
            m.load_state_dict(torch.load(config.data_name+"model"+str(i)+".pt"))
    print(m, file=open('log.txt', 'a'))
    input_embd.load_state_dict(torch.load(config.data_name+"_embd.pt"))
else:
    # Create Train Obj and train all AEs step by step
    train_obj = TrainClass(ae_list)
    train_obj.TrainAll(train_DataLoader_FD, input_embd, loss_fn)
    torch.save(input_embd.state_dict(), config.data_name+"_embd.pt")



# Create the classifier
classifier = Classifier(config.hidden_dims, ae_list, input_embd)

if torch.cuda.is_available():
    classifier = classifier.cuda()

# Define the optimizer and loss function
optimizer = torch.optim.Adam(classifier.parameters(), lr = 0.001)
loss_function = nn.BCELoss()


# print(ae_list[1].encoder_block.rnn.weight_ih_l0, file=open('log.txt', 'a'))

kfolds= 5
kfold= KFold(n_splits= kfolds, shuffle= True)


# K fold val loop
for fold, (train_idx, val_idx) in enumerate(kfold.split(train_ds),1):
    print("____________ Train Fold No-{}___________".format(fold), file=open("log.txt", 'a'))
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
        train(fold, train_dataloader, epoch, classifier, optimizer, loss_function)
    print("\n\nValidate  \n\n", file=open("log.txt", 'a'))
    validate(fold, val_dataloader, classifier, optimizer, loss_function)
    
    print("\n\nTest  \n\n", file=open("log.txt", 'a'))
    test(test_dataloader, classifier) 

    classifier.apply(reset_weights)

    
    # reinitialize the encoders weights
    classifier.embedding.load_state_dict(torch.load(config.data_name+"_embd.pt"))

    # reinitialize the encoders weights
    for i, m in enumerate(ae_list):
            m.load_state_dict(torch.load(config.data_name+"model"+str(i)+".pt"))

    classifier.encoder1 = ae_list[0].encoder_block
    classifier.encoder2 = ae_list[1].encoder_block
    classifier.encoder3 = ae_list[2].encoder_block

