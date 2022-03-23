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
import pandas as pd

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode")
    parser.add_argument("--checkpoint", default=argparse.SUPPRESS)
    parser.add_argument("--use_extra_ids", action="store_true")
    parser.add_argument("--use_extra_data", action="store_true")
    parser.add_argument("--new_turtles_fraq", default=0., type=float)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--views_model", default=argparse.SUPPRESS)
    parser.add_argument("--name", default="turtle", type=str)

    args = parser.parse_args()

    if args.wandb:
        use_wandb = True

    load_images()
    train, extra, test = load_csv()
  
    if args.new_turtles_fraq > 0.:
        n_unknown = int(train.shape[0]*args.new_turtles_fraq)
        unknown = extra[~extra["is_known_id"]].sample(n_unknown)
        unknown["turtle_id"] = "new_turtle"
        train = pd.concat((train, unknown))
        print(f"Add {n_unknown} new_turtles")
    elif args.use_extra_ids:
        train = pd.concat((train, extra))
    if args.use_extra_data:
        train = pd.concat((train, extra[extra.is_known_id]))
    
    idx2id = train["turtle_id"].unique()
    id2idx = {v : i for i, v in enumerate(idx2id)}
    torch.save(idx2id, "idx2id.pt")

    train, val = train_val_split(train, train_val_split_fraq, True)
    print(train.shape, val.shape, test.shape)

    views_model = None
    model = get_model(num_classes, device, model_type)
    if "checkpoint" in args:
        model.load_state_dict(torch.load(args.checkpoint))
    if "views_model" in args:
        views_model = get_model(3, device, "simple")
        views_model.load_state_dict(torch.load(args.views_model))

    if args.mode == "train":
        dataloaders_dict = {
            "train" : get_dataloader(train, train_transforms, id2idx, batch_size),
            "val" : get_dataloader(val, val_transforms, id2idx, batch_size)
        }

        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(model.parameters(), lr=3e-4)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001) 
        scheduler = None

        model = train_model(
            model, dataloaders_dict, criterion, optimizer,
            scheduler=scheduler, num_epochs=num_epochs,
            device=device, use_wandb=use_wandb, mode=model_type,
            target="labels", name="model", views_model=views_model,
            wandb_name=args.name
        )
    elif args.mode == "train_views":
        dataloaders_dict = {
            "train" : get_dataloader(train, train_transforms, id2idx, batch_size),
            "val" : get_dataloader(val, val_transforms, id2idx, batch_size)
        }

        model = get_model(3, device, "simple")        

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = None

        model = train_model(
            model, dataloaders_dict, criterion, optimizer,
            scheduler=scheduler, num_epochs=2,
            device=device, use_wandb=use_wandb, mode="simple",
            target="views", name="model_views"
        )    
    elif args.mode == "byol":
        dataloaders_dict = {
            "train" : get_dataloader(train, byol_transforms, id2idx, batch_size)
        }
        model = train_byol(model, dataloaders_dict)
    elif args.mode == "eval":
        dataloaders_dict = {
            "train" : get_dataloader(
                        train,
                        train_transforms,
                        id2idx,
                        batch_size
                    ),
              "val" : get_dataloader(
                        val,
                        val_transforms,
                        id2idx,
                        batch_size
                    )
        }

        criterion = nn.CrossEntropyLoss()

        eval_model(model, dataloaders_dict, criterion)