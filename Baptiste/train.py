import time
from timeit import default_timer as timer
from torchvision import transforms
import torch
import copy 

import torch.nn as nn



def train_model(model, optimizer, num_epochs,data_loaders, patience=5):
    since = time.time()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    consecutive_epochs_no_improvement = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    dataset_sizes = {phase: len(data_loaders[phase].dataset) for phase in ['train', 'val']}

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        running_loss = {'train': 0.0, 'val': 0.0}

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for inputs, targets in data_loaders[phase]:
                pixel_values = inputs.to('cuda:0')
                labels = targets.to('cuda:0')
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    loss, logits = outputs.loss, outputs.logits
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss[phase] += loss.item() * inputs.size(0)

            epoch_loss = running_loss[phase] / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            if phase == 'train':
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)

                # Check for early stopping
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    consecutive_epochs_no_improvement = 0
                else:
                    consecutive_epochs_no_improvement += 1

        if consecutive_epochs_no_improvement >= patience:
            print(f'Early stopping after {patience} consecutive epochs without improvement.')
            break

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    model.load_state_dict(best_model_wts)
    return train_losses, val_losses, model
