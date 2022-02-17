import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from model import get_model
import numpy as np
from src.data import load_csv, load_images, get_dataloader
from src.train import train_model

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

SEED = 424242
torch.manual_seed(SEED)
np.random.seed(SEED)

num_classes = 100
batch_size = 8
num_epochs = 15
feature_extract = False
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":

    load_images()
    train, test = load_csv()

    dataloaders_dict = {
        "train" : get_dataloader(train, )
    }

    model = get_model(num_classes, device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = None
    criterion = nn.CrossEntropyLoss()
    
    model_ft = train_model(
        model, dataloaders_dict, criterion, optimizer,
        scheduler=scheduler, num_epochs=num_epochs,
        device=num_epochs, use_wandb=False
    )