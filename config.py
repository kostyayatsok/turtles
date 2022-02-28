import torch 
from torchvision import transforms
import os
import numpy as np

SEED = 424242

torch.manual_seed(SEED)
np.random.seed(SEED)

num_classes = 100
batch_size = 8
num_epochs = 20
input_size = 224
train_val_split_fraq = 0.9
feature_extract = False
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

IMAGE_DIR = './turtle_recall/images'

CHECKPOINTS_DIR = "./checkpoints/"
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

train_transforms = transforms.Compose([
    transforms.Resize(input_size),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize(input_size),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

byol_transforms = transforms.Compose([
    transforms.Resize(4*input_size),
])

byol_epochs = 5