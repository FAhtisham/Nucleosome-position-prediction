from utils import convert_seqs_to_words, add_padding, ngram_counter

from sklearn.utils import shuffle
import numpy as np
import csv

import config

import torch
from torch.utils.data import TensorDataset, DataLoader

def read_csv(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    print("Total Samples in the dataset: ", len(data))
    return data



def preprocess_data(data, ngram):
    samples = []
    labels = []

    # Separate the samples and their labels
    for i in range(len(data)):
        samples.append(data[i][0])
        labels.append(data[i][1])


    # convert sequences into trigrams of nucleotides 
    samples = convert_seqs_to_words(samples, ngram)

    print("Creating trigrams ...") 

    # Create encoding and decoding dictionaries
    nuc_pairs= ngram_counter(samples)

    encoding= {w:i for i,w in enumerate(nuc_pairs,0)}
    decoding= {i:w for i,w in enumerate(nuc_pairs,0)}


    # Encode the trigrams 
    for i in range(len(samples)):
        temp = []
        seq = samples[i]
        for k in range(len(seq)):
            temp.append(encoding[seq[k]])
        samples[i] = temp
    
    # Create the numpy arrays 
    samples_np = np.asarray(samples,  dtype=np.int64)
    labels_np = np.asarray(labels, dtype = np.int64)

    # Way to create a smaller subset of the total dataset
    # samples_np_f = samples_np[:250]
    # samples_np_l = samples_np[samples_np.shape[0]-250:-1]


    # labels_np_f = labels_np[:250]
    # labels_np_l = labels_np[labels_np.shape[0]-250:-1]



    # samples_np = np.concatenate((samples_np_f, samples_np_l), axis=0)
    # labels_np = np.concatenate((labels_np_f, labels_np_l), axis=0)

    print("\nNumpy arrays of the oringinal dataset have been created ....")
    return samples_np, labels_np, samples, encoding, decoding

# Function to create and return the loaders
def create_loaders(samples_np, labels_np, samples, split_size):

    '''
        args:
            split_size: refers to the amount of training dataset
    '''
    full_dataset = TensorDataset(torch.from_numpy(samples_np), torch.Tensor(labels_np))
    samples_np, labels_np = shuffle(samples_np, labels_np, random_state=0)
    
    train_ds, test_ds = torch.utils.data.random_split(full_dataset, [round((len(samples)*split_size)/100), len(samples)-round((len(samples)*split_size)/100)])
 
    train_DataLoader = DataLoader(train_ds, batch_size = config.batch_size, shuffle = True, sampler = None,
                            batch_sampler = None, num_workers = 0, collate_fn = None,
                            pin_memory = False, drop_last = False, timeout = 0, worker_init_fn = None)

    test_DataLoader = DataLoader(test_ds, batch_size = config.batch_size, shuffle = False, sampler = None,
                            batch_sampler = None, num_workers = 0, collate_fn = None,
                            pin_memory = False, drop_last = False, timeout = 0, worker_init_fn = None)

    return train_DataLoader, test_DataLoader, train_ds, test_ds, full_dataset