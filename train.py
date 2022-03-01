import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
from src.model import get_model
from src.data import load_csv, load_images, get_dataloader, train_val_split
from src.train import train_model
from src.eval import eval_model
from src.byol import train_byol
from config import *
import argparse

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode")
    parser.add_argument("--checkpoint")

    args = parser.parse_args()

    load_images()
    train, test = load_csv()

    idx2id = train["turtle_id"].unique()
    id2idx = {v : i for i, v in enumerate(idx2id)}

    train, val = train_val_split(train, train_val_split_fraq)
    

    model = get_model(num_classes, device)
    if "checkpoint" in args:
        model.load_state_dict(torch.load(args.checkpoint))
        model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = None

    if args.mode == "train":
        dataloaders_dict = {
            "train" : get_dataloader(train[train.is_known_id], train_transforms, id2idx, batch_size),
            "val" : get_dataloader(val[val.is_known_id], val_transforms, id2idx, batch_size)
        }

        criterion = nn.CrossEntropyLoss()

        model = train_model(
            model, dataloaders_dict, criterion, optimizer,
            scheduler=scheduler, num_epochs=num_epochs,
            device=device, use_wandb=use_wandb
        )
    elif args.mode == "byol":
        dataloaders_dict = {
            "train" : get_dataloader(train, byol_transforms, id2idx, batch_size)
        }
        model = train_byol(model, dataloaders_dict)
    elif args.mode == "eval":
        dataloaders_dict = {
            "train" : get_dataloader(
                        train[train.is_known_id],
                        train_transforms,
                        id2idx,
                        batch_size
                    ),
              "val" : get_dataloader(
                        val[val.is_known_id],
                        val_transforms,
                        id2idx,
                        batch_size
                    )
        }

        criterion = nn.CrossEntropyLoss()

        eval_model(model, dataloaders_dict, criterion)