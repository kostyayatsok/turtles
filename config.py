import torch 
from torchvision import transforms

SEED = 424242
num_classes = 100
batch_size = 8
num_epochs = 15
input_size = 224
feature_extract = False
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
IMAGE_DIR = './turtle_recall/images'


train_transforms = transforms.Compose([
    transforms.Resize(input_size),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize(input_size),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])