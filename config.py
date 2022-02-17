import torch 

SEED = 424242
num_classes = 100
batch_size = 8
num_epochs = 15
feature_extract = False
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
IMAGE_DIR = './turtle_recall/images'