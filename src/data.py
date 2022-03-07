import io
import numpy as np
import os
import pandas as pd
import requests
import urllib.parse

from src.TurtleDataset import TurtleDataset
from torch.utils.data import DataLoader

from config import IMAGE_DIR

def load_images():
    SOURCE_URL = 'https://storage.googleapis.com/dm-turtle-recall/images.tar'
    TAR_PATH = os.path.join(IMAGE_DIR, os.path.basename(SOURCE_URL))
    EXPECTED_IMAGE_COUNT = 13891

    os.system(f"mkdir --parents {IMAGE_DIR}")
    if len(os.listdir(IMAGE_DIR)) != EXPECTED_IMAGE_COUNT:
        os.system(f"wget --no-check-certificate -O {TAR_PATH} {SOURCE_URL}")
        os.system(f"tar --extract --file={TAR_PATH} --directory={IMAGE_DIR}")
        os.system(f"rm {TAR_PATH}")

    print(f'The total number of images is: {len(os.listdir(IMAGE_DIR))}')

def read_csv_from_web(file_name):
    BASE_URL = 'https://storage.googleapis.com/dm-turtle-recall/'
    url = urllib.parse.urljoin(BASE_URL, file_name)
    content = requests.get(url).content
    return pd.read_csv(io.StringIO(content.decode('utf-8')))

def load_csv(use_extra=True):
    # Read in csv files.
    train = read_csv_from_web('train.csv')
    test = read_csv_from_web('test.csv')
    
    all_ids = np.unique(train.turtle_id)
    if use_extra:
        extra = read_csv_from_web('extra_images.csv')
        train = pd.concat((train, extra))
    
    train["is_known_id"] = train["turtle_id"].isin(all_ids)

    return train, test

def train_val_split(data, train_frac=0.7, shuffle=False):
    train_size = int(data.shape[0]*train_frac)
    if shuffle:
        data = data.sample(data.shape[0])
    train, val = data[:train_size], data[train_size:]
    print(f"Using {train.shape} images for training and {val.shape} images for validation")
    return train, val

def get_dataloader(
    data, data_transforms, id2idx, batch_size=8
):
    # Create training and validation datasets
    image_dataset = TurtleDataset(data, data_transforms, id2idx)
    # Create training and validation dataloaders
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    return dataloader