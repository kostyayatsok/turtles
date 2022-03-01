import torch
from tqdm.auto import tqdm
from config import device

@torch.no_grad()
def eval_model(model, dataloaders, criterion):    
    for phase, loader in dataloaders.items():
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(
                                dataloaders[phase],
                                total=len(dataloaders[phase])):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Get model outputs and calculate loss
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        loss = running_loss / len(dataloaders[phase].dataset)
        acc = running_corrects.double() / len(dataloaders[phase].dataset)

        print('Phase {} Loss: {:.4f} Acc: {:.4f}'.format(phase, loss, acc))