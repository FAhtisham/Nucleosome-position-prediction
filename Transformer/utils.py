import copy
import numpy as np
from tqdm import tqdm
import collections

# Padding function
def add_padding(seq, p_len):
  seq = seq + ("P" * p_len)
  return seq
  
# Convert nucleotides to 3-grams
def convert_seqs_to_words(sequences, ngram):
  f_sequences = []
  for i in range(len(sequences)):
    temp = ""
    str_ = sequences[i]
    j=0
    if len(str_)%ngram!=0:
      n = len(str_)
      while n % ngram != 0:
        n+=1
        str_= add_padding(str_,n-len(str_))
    for k in range(0,len(str_)):
      j+=1
      if  j%ngram==0:
        temp = temp + str_[k-ngram+1:j] + ' '   
      #j+=3
    f_sequences.append(temp) 
  f_sequences= [j.split() for j in f_sequences]
  return f_sequences

# ngram encoder and decoder function
def ngram_counter(seqs, clip=1):
  nuc_pairs= collections.Counter()

  for seq in tqdm(seqs):
    nuc_pairs.update(seq)

  # check why this statement is so important (84, remains same without it)
  for nucs in list(nuc_pairs.keys()):
    if nuc_pairs[nucs] < clip:
      nuc_pairs.pop(nucs)

  return list(sorted(nuc_pairs.keys()))


# Load the pretrained embedding file
def load_pretrained_embd(File):
    print("Loading DNA2vec Embeddings ......")
    embd_model = {}
    with open(File,'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            embd_model[word] = embedding
    print(f"{len(embd_model)} words loaded!")
    return embd_model


# Create the embedding matrix 
def create_embd_matrix(wi, embd_dict, dimension):
  embd_matrix = np.zeros((len(wi)+1, dimension))

  for w, i in wi.items():
    if w in embd_dict:
      embd_matrix[i] = embd_dict[w]
  return embd_matrix

# embd_matrix = create_embd_matrix(encoding, embd_dict, 100)
# embd_matrix.shape


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])