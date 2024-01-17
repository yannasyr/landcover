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
        model,optimizer,model_name=segformer(lr=0.0001)
        
    elif args.unet :
        model = UNet2(4, 10, bilinear=False)
        if args.test:
            pretrained_state_dict = torch.load("Unet_epoch85.pt")
            model.load_state_dict(pretrained_state_dict)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        model_name ="Unet"
    # move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ##dataset and dataloader
    # Définir le chemin du dossier d'entraînement
    train_data_folder = 'train'

    # Créer un objet Dataset pour l'ensemble d'entraînement
    train_dataset = LandscapeData(train_data_folder, transform=data_transforms['train'])
    print(len(train_dataset))

    # Diviser l'ensemble d'entraînement en ensembles d'entraînement, de validation et de test (80% - 10% - 10%)
    train_size = int(0.8 * len(train_dataset))
    val_test_size = len(train_dataset) - train_size
    if val_test_size % 2 == 0:
        val_size = test_size = val_test_size // 2
    else:
        val_size = (val_test_size - 1) // 2
        test_size = val_test_size - val_size

    train_dataset, val_test_dataset = random_split(train_dataset, [train_size, val_test_size], generator=torch.Generator().manual_seed(42))
    val_dataset, test_dataset = random_split(val_test_dataset, [val_size, test_size], generator=torch.Generator().manual_seed(42))

    # Créer des DataLoader pour les ensembles d'entraînement, de validation et de test
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
    data_loaders = {'train': train_loader, 'val': val_loader}

    # Number of images in train and val sets
    num_train_images = len(train_dataset)
    num_val_images = len(val_dataset)
    num_test_images = len(test_dataset)

    print(f"Number of images in the training set: {num_train_images}")
    print(f"Number of images in the validation set: {num_val_images}")
    print(f"Number of images in the validation set: {num_test_images}")
    
    #hyperparametres
    Num_epoch=200
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)

    ##training
    if args.train :
        train_losses,val_losses , model  = train_model(model,model_name, optimizer,scheduler,  Num_epoch,data_loaders)
        dataloader_metrics=val_loader
    if args.test :
        dataloader_metrics=test_loader


    ##Eval on val loader or test loader -> args test or train 
    print(mesure_on_dataloader(dataloader_metrics,device,model))
    mean_iou, mean_accuracy, per_category_iou, Overall_acc = compute_average_metrics(model, dataloader_metrics,classes_to_ignore=args.classes_to_ignore)
    print("Mean_iou:", mean_iou)
    print("Mean accuracy:", mean_accuracy)
    print("IoU per category", per_category_iou)
    print("OA", Overall_acc)

    #affichage(model,val_loader,device)



    # Ouvrir un fichier texte en mode écriture
    with open('losses.txt', 'w') as file:
        # Écrire les données de perte d'entraînement
        file.write("Train Losses:\n")
        for loss in train_losses:
            file.write(str(loss) + '\n')

        # Écrire les données de perte de validation
        file.write("\nValidation Losses:\n")
        for loss in val_losses:
            file.write(str(loss) + '\n')

    print("Les pertes ont été enregistrées dans le fichier 'losses.txt'.")




