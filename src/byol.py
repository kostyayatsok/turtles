import torch
from byol_pytorch import BYOL
from torchvision import models
import time
from config import *
from tqdm.auto import tqdm

def train_byol(model, dataloaders):

    # resnet = models.resnet50(pretrained=True)
    since = time.time()
    learner = BYOL(
        model,
        image_size = input_size,
        hidden_layer = 'avgpool',
        use_momentum=False
    )
    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

    byol_epochs = 5
    for epoch in range(byol_epochs):
        print('Epoch {}/{}'.format(epoch+1, byol_epochs))
        print('-' * 10)
        phase = 'train'
        running_loss = 0.0
        
        # Iterate over data.
        for i, (inputs, labels) in tqdm(
                                    enumerate(dataloaders[phase]),
                                    total=len(dataloaders[phase])):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.set_grad_enabled(phase == 'train'):
                if phase == 'train':
                    loss = learner(inputs)
                    if loss.isnan():
                        raise "====>>>>> ooops loss is nan <<<<<====="
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    running_loss+=loss.item()*batch_size
            if i and i % 150 == 0:
                print('\n{} Loss: {:.4f}'.format(phase, running_loss/(i*batch_size)))
        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        torch.save(model.state_dict(), f'{CHECKPOINTS_DIR}/improved-net.pt')

        print('{} Loss: {:.4f}'.format(phase, epoch_loss))

        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))

    return model