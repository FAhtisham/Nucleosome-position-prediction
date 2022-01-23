import logging

dataset_path = "/netscratch/abbasi/nucleosome_prediction/group1/Homo_Sapiens.csv"
embd_file_path = "/netscratch/abbasi/nucleosome_prediction/dna.txt"

use_pretrained_embd  = True
embedding_dims = 100
e_hidden_dims = 200
d_model = embedding_dims
finetune= False
use_gpu = True
use_attention = True


dropout_size = 0.2
epochs = 500
batch_size=64
ngram=4


log_file = 'logM'

# Create the file logger
logging.basicConfig(level=logging.DEBUG, filename="logfileM", filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")
logger = logging