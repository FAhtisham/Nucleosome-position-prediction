dataset_path = "/netscratch/abbasi/nucleosome_prediction/group1/Homo_Sapiens.csv"
embd_file_path = "/netscratch/abbasi/nucleosome_prediction/dna.txt"
data_list=["HSG1","DSG1", "ELG1", "YG1"]

data_name = data_list[0]

use_pretrained_embd  = True
embedding_dims = 100
hidden_dims = 100
out_dims = hidden_dims

finetune=True
use_gpu = True

dropout_size = 0.2
epochs = 9
batch_size=32
ngram=8
nlayers=1



use_pretrained_aes = True