import copy
import time
import torch
from tqdm.auto import tqdm

from config import CHECKPOINTS_DIR

def train_model(
    model, dataloaders, criterion, optimizer,
    scheduler=None, num_epochs=25, device='cpu', use_wandb=False,
    mode="simple", target="labels", name="model", views_model=None
):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    if use_wandb:
        import wandb
        wandb.init(project="turtles")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, views in tqdm(
                                    dataloaders[phase],
                                    total=len(dataloaders[phase])):
                if target == "views":
                    labels = views
                optimizer.zero_grad()

                outputs, preds = predict(
                    model, inputs, labels, views,
                    device, phase, views_model, mode
                )
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if use_wandb:
                wandb.log({
                    "epoch":epoch,
                    f"loss_{phase}":epoch_loss,
                    f"acc_{phase}":epoch_acc,
                })
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), f"{CHECKPOINTS_DIR}/{name}.pt")
        print()
        if scheduler is not None:
            scheduler.step()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def predict(
    model, inputs, labels, views, device,
    phase='val', views_model=None, mode='simple'
):
    inputs = inputs.to(device)
    labels = labels.to(device)
    views = views.to(device).float()
    if views_model is not None:
        _, views_p = torch.max(views_model(inputs), dim=1)
        mask = (views == views)
        views[~mask] = views_p[~mask].float()

    # zero the parameter gradients

    # forward
    # track history if only in train
    with torch.set_grad_enabled(phase == 'train'):
        # Get model outputs and calculate loss
        if mode == "simple":
            outputs = model(inputs)
        elif mode == "multihead":
            outputs = model(inputs, views)

        _, preds = torch.max(outputs, 1)

    return outputs, preds