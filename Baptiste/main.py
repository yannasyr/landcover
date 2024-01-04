#!C:\Users\BABA\AppData\Local\Programs\Python\Python310\python.exe
import torchvision.transforms as transforms
from Landscapedata import LandscapeData
from train import train_model
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from metrics import mesure_on_dataloader , affichage , compute_average_metrics
import os 
from models import segformer, UNet2
from arg_parser import parser
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter



if __name__ == "__main__":

    args = parser()
    means =  [ 378.48734924,  631.65566376,  531.96720581, 3500.04284851]
    stds =  [294.72591834, 359.42661458, 488.99842265, 752.03863059]
    # Specify the mean and std for each channel in the transforms.Normalize
    cur_means = means[:4]  
    cur_stds = stds[:4]    
    # Utilisez les bonnes transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cur_means, cur_stds),
        ])
    }
    if args.segformer :
        model,optimizer=segformer()
    elif args.unet :
        model = UNet2(4, 10, bilinear=False).to('cuda:0')
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

    data_folder = "small_dataset"
    dataset = LandscapeData(data_folder, transform=data_transforms['train'])  # Utilisez la transformation 'train'
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    data_loaders = {'train': train_loader, 'val': val_loader}


    images, masks = next(iter(train_loader))
    print("Image : ", images.shape)
    print("masks : ", masks.shape)

    #Initialize a Counter to count class frequencies
    class_counter = Counter()

    # Iterate over the training dataset to count class occurrences
    for _, targets in train_loader:
        class_counter.update(targets.flatten().numpy())

    # Calculate proportions
    total_samples = sum(class_counter.values())
    class_proportions = {class_idx: count / total_samples for class_idx, count in class_counter.items()}

    # Display class indices and proportions
    for class_idx, proportion in class_proportions.items():
        print(f"Class {class_idx}: Proportion = {proportion:.4f}")



    # move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model.to(device)

    Num_epoch=100
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1)

    train_losses,val_losses , model  = train_model(model, optimizer,scheduler,  Num_epoch,data_loaders)

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    #affichage(model,val_loader,device)
    #print(mesure_on_dataloader(val_loader,device,model))
    mean_iou, mean_accuracy, per_category_iou = compute_average_metrics(model, val_loader,classes_to_ignore=[0,1,7,8,9])

    print("Mean_iou:", mean_iou)
    print("Mean accuracy:", mean_accuracy)
    print("IoU per category", per_category_iou)


    
    # # save best model 
    # print("Saving best model...")
    # if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    # save_point = os.path.join("checkpoint", )
    # torch.save(model.state_dict(), save_point + '.pt')
    # print("Model saved!")


##SegFormer##
#     Mean_iou: 0.5198857023384602
# Mean accuracy: 0.6122436394994278
# IoU per category [       nan 0.         0.66164696 0.74257229 0.76370562 0.67365327
#  0.64458844 0.38187744 0.         0.8109273 ]

##Unet##
# Mean_iou: 0.488395057931877
# Mean accuracy: 0.5732042679625972
# IoU per category [       nan 0.         0.68719141 0.78629205 0.78929071 0.6984251
#  0.68911637 0.         0.         0.74523988]

