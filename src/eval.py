import torch
from tqdm.auto import tqdm
from config import device
from metrics import mapk

@torch.no_grad()
def eval_model(model, dataloaders, criterion):    
    for phase, loader in dataloaders.items():
        running_loss = 0.0
        running_corrects = 0

        predictions, labels = [], []
        for inputs, labels in tqdm(
                                dataloaders[phase],
                                total=len(dataloaders[phase])):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Get model outputs and calculate loss
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            _, top5 = torch.topk(outputs, 5)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            labels.append(labels)
            predictions.append(top5)
        labels = torch.hstack(labels)
        predictions = torch.vstack(predictions)

        loss = running_loss / len(dataloaders[phase].dataset)
        acc = running_corrects.double() / len(dataloaders[phase].dataset)
        map5 = mapk(labels, predictions, 5)
        map1 = mapk(labels, predictions, 1)


        print('Phase {} Loss: {:.4f} Acc: {:.4f} Map5: {:.4f} Map1: {:.4f}'.format(phase, loss, acc, map5, map1))