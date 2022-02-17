from torchvision import models
import torch.nn as nn

def get_model(num_classes, device='cpu'):
    model_ft = models.resnet50(pretrained=True).to(device)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft