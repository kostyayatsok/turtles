import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
from src.model import get_model
from src.data import load_csv, load_images, get_dataloader, train_val_split
from src.train import train_model
from src.byol import train_byol
from config import *
import sys
import os

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

if __name__ == "__main__":

    load_images()
    train, test = load_csv()

    idx2id = train["turtle_id"].unique()
    id2idx = {v : i for i, v in enumerate(idx2id)}

    train, val = train_val_split(train, 0.9)

    dataloaders_dict = {
        "train" : get_dataloader(train, train_transforms, id2idx, batch_size),
        "val" : get_dataloader(val, val_transforms, id2idx, batch_size)
    }

    model = get_model(num_classes, device)

    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = None

    if sys.argv[1] == "train":
        criterion = nn.CrossEntropyLoss()
        
        model_ft = train_model(
            model, dataloaders_dict, criterion, optimizer,
            scheduler=scheduler, num_epochs=num_epochs,
            device=device, use_wandb=False
        )
    elif sys.argv[1] == "byol":
        resnet = train_byol(model, dataloaders_dict)