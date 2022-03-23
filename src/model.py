from torchvision import models
import torch.nn as nn
import torch

def get_model(num_classes, device='cpu', model_type="simple"):
    if model_type == "simple":
        # model = models.resnet18(pretrained=True)
        # num_ftrs = model.fc.in_features
        # model.fc = nn.Linear(num_ftrs, num_classes)

        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)

        # model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        # num_ftrs = model.classifier.fc.in_features
        # model.classifier.fc = nn.Linear(num_ftrs, num_classes)
    elif model_type == "multihead":
        model = MultiheadModel(num_classes)
    return model.to(device)

class MultiheadModel(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        # model = models.resnet50(pretrained=True)
        # num_ftrs = model.fc.in_features
        # model.fc = nn.Linear(num_ftrs, 3*num_classes)
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
          nn.Linear(num_ftrs, num_classes),
          nn.Linear(num_classes, 3*num_classes)
        )

        self.model = model
        self.num_classes = num_classes

    def forward(self, image, view):
        out = self.model(image)

        n = self.num_classes
        mask = view.view(-1, 1).repeat(1, 3*n)
        mask[:,0*n:1*n] = mask[:,0*n:1*n] == 0
        mask[:,1*n:2*n] = mask[:,1*n:2*n] == 1
        mask[:,2*n:3*n] = mask[:,2*n:3*n] == 2
        mask = mask.bool()
        out = out[mask].view(-1, n)

        return out

if __name__ == "__main__":
    model = MultiheadModel(10)
    model(torch.zeros(8, 3, 224, 224), torch.zeros(8))