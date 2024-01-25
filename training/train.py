import time
from timeit import default_timer as timer
from torchvision import transforms
import torch
import copy 
from arg_parser import parser
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from LandCoverData import LandCoverData as LCD
import torch.nn.functional as F

args = parser()



def train_model(model,model_name, optimizer,scheduler, num_epochs,data_loaders, patience=5):
    since = time.time()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    consecutive_epochs_no_improvement = 0
    best_model_wts = copy.deepcopy(model.state_dict())


    # Supposez que `y_true` et `y_pred` soient vos vraies étiquettes et prédictions respectives
    # `y_true` doit être un tenseur contenant les indices de classe (entiers)
    # `y_pred` doit contenir les logits avant l'application de softmax


    if args.weighted :
        # note: we set to 0 the weights for the classes "no_data"(0), "clouds"(1) and "snow"(8) to ignore these
        class_weight = (1 / LCD.TRAIN_CLASS_COUNTS[:])* LCD.TRAIN_CLASS_COUNTS[:].sum() / (LCD.N_CLASSES)
        class_weight[LCD.IGNORED_CLASSES_IDX] = 0.


        # Convertir en torch.Tensor
        class_weight = torch.tensor(class_weight, dtype=torch.float32)

        # Transférer sur CUDA si disponible
        if torch.cuda.is_available():
            class_weight = class_weight.cuda()

        print(f"Will use class weights: {class_weight}")
        criterion = nn.CrossEntropyLoss(weight=class_weight)
        
        
    else :
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        
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
                    if args.unet :
                        outputs = model(pixel_values)
                        logits = outputs
                        loss = criterion(logits, labels.squeeze(dim=1))

                    elif args.segformer  : 
                        outputs = model(pixel_values=pixel_values, labels=labels)
                        logits =outputs.logits
                        upsampled_logits = nn.functional.interpolate(
                        logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                        loss=criterion(upsampled_logits,labels)

                    elif args.deeplab :
                        outputs = model(pixel_values)['out']
                        logits=outputs
                        labels = labels.squeeze(1)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss[phase] += loss.item() * inputs.size(0)

            epoch_loss = running_loss[phase] / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            if phase == 'val':
                scheduler.step(epoch_loss)  # Step the scheduler on validation loss


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
        # Save the model every 5 epochs
        if epoch % 5 == 0 and args.save_model:
            print("Saving model at epoch {}...".format(epoch))
            save_point = os.path.join("checkpoint", f"{model_name}_epoch{epoch}")
            torch.save(model.state_dict(), save_point + '.pt')
            print("Model saved!")


    time_elapsed = time.time() - since
    print("Entraînement terminé en {:.0f}h {:.0f}m {:.0f}s".format(time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))
    model.load_state_dict(best_model_wts)
    return train_losses, val_losses, model
