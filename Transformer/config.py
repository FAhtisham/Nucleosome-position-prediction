import logging

dataset_path = "/netscratch/abbasi/nucleosome_prediction/group1/Homo_Sapiens.csv"
embd_file_path = "/netscratch/abbasi/nucleosome_prediction/dna.txt"

use_pretrained_embd  = True
embedding_dims = 100
e_hidden_dims = 200
d_model = embedding_dims

dropout_size = 0.2
epochs = 30
batch_size=64
ngram=8


attn_heads=2


log_file = 'logM'

# Create the file logger
logging.basicConfig(level=logging.DEBUG, filename=log_file, filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")
logger = logging