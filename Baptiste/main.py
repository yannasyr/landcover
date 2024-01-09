import torchvision.transforms as transforms
from Landscapedata import LandscapeData
from train import train_model
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
from metrics import mesure_on_dataloader , affichage , compute_average_metrics
import os 
from models import segformer, UNet2
from arg_parser import parser
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter
import torch.backends.cudnn as cudnn


if __name__ == "__main__":
    cudnn.benchmark = True
    args = parser()
    ##Normalisation 
    means =  [ 418.19976217,  703.34810956,  663.22678147, 3253.46844222]
    stds =  [294.73191962, 351.31328415, 484.47475774, 793.73928079]

    ##transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    }

    ##model selection
    if args.segformer :
        model,optimizer,model_name=segformer()
    elif args.unet :
        model = UNet2(4, 10, bilinear=False)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        model_name ="Unet"
    # move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ##dataset and dataloader
    data_folder = "datasetV2\\main\\train"
    dataset = LandscapeData(data_folder, transform=data_transforms['train']) 
    print(len(dataset))
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    data_loaders = {'train': train_loader, 'val': val_loader}

    # Number of images in train and val sets
    num_train_images = len(train_dataset)
    num_val_images = len(val_dataset)

    print(f"Number of images in the training set: {num_train_images}")
    print(f"Number of images in the validation set: {num_val_images}")

    #hyperparametres
    Num_epoch=200
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1)

    ##training
    train_losses,val_losses , model  = train_model(model, optimizer,scheduler,  Num_epoch,data_loaders)

    ##Eval on val loader 
    print(mesure_on_dataloader(val_loader,device,model))
    mean_iou, mean_accuracy, per_category_iou, Overall_acc = compute_average_metrics(model, val_loader,classes_to_ignore=args.classes_to_ignore)
    print("Mean_iou:", mean_iou)
    print("Mean accuracy:", mean_accuracy)
    print("IoU per category", per_category_iou)
    print("OA", Overall_acc)
    #affichage(model,val_loader,device)

    ##plot losses
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # # save best model 
    if args.save_model :
        print("Saving best model...")
        save_point = os.path.join("checkpoint", f"{model_name}")
        torch.save(model.state_dict(), save_point + '.pt')
        print("Model saved!")







##SegFormer##
#     Mean_iou: 0.5198857023384602
# Mean accuracy: 0.6122436394994278
# IoU per category [       nan 0.         0.66164696 0.74257229 0.76370562 0.67365327
#  0.64458844 0.38187744 0.         0.8109273 ]


##segFormer## classes exclu : [0,1,7,8,9]


##Unet##
# Mean_iou: 0.488395057931877
# Mean accuracy: 0.5732042679625972
# IoU per category [       nan 0.         0.68719141 0.78629205 0.78929071 0.6984251
#  0.68911637 0.         0.         0.74523988]

