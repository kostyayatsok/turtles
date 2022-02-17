import torch
from torchvision.io import read_image
from config import IMAGE_DIR

class TurtleDataset(torch.utils.data.Dataset):
    def __init__(self, df, transforms, id2idx):
        self.df = df
        self.transforms = transforms
        self.id2idx = id2idx
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, idx):
        img = read_image(f"{IMAGE_DIR}/{self.df.iloc[idx]['image_id']}.JPG") / 255.
        pad_h = max(img.size(1), img.size(2)) - img.size(1)
        pad_w = max(img.size(1), img.size(2)) - img.size(2)
        img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h, 0, 0))
        img = self.transforms(img.float())
        
        label = self.id2idx[self.df.iloc[idx]['turtle_id']]
        return img, label