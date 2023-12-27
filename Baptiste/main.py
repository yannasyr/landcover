import torchvision.transforms as transforms
from Landscapedata import LandscapeData
from train import train_model
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from metrics import mesure_on_dataloader , affichage
import os 
from models import segformer
from arg_parser import parser




if __name__ == "__main__":
  # Exemple d'utilisation
  means =  [ 378.48734924,  631.65566376,  531.96720581, 3500.04284851]

  stds =  [294.72591834, 359.42661458, 488.99842265, 752.03863059]
  # Specify the mean and std for each channel in the transforms.Normalize
  cur_means = means[:4]  # Take the first 4 values
  cur_stds = stds[:4]    # Take the first 4 values

  # Utilisez les bonnes transformations
  data_transforms = {
      'train': transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(cur_means, cur_stds),
      ])
  }

  args = parser()
  if args.segformer :
      model,optimizer=segformer()
  elif args.unet :
      model , optimizer = None

  data_folder = "dataset//train"
  dataset = LandscapeData(data_folder, transform=data_transforms['train'])  # Utilisez la transformation 'train'
  train_size = int(0.8 * len(dataset))
  val_size = len(dataset) - train_size
  train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

  train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

  data_loaders = {'train': train_loader, 'val': val_loader}


  images, masks = next(iter(train_loader))
  print("Image : ", images.shape)
  print("masks : ", masks.shape)

  # move model to GPU
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  if args.segformer :
      model,optimizer=segformer()
  elif args.unet :
      model , optimizer = None


  model.to(device)

  Num_epoch=100

  train_losses,val_losses , model  = train_model(model, optimizer,  Num_epoch,data_loaders)

  plt.plot(train_losses, label='Train Loss')
  plt.plot(val_losses, label='Validation Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()
  affichage(model,val_loader,device)
  print(mesure_on_dataloader(val_loader,device,model))

    # save best model 
  print("Saving best model...")
  if not os.path.isdir('checkpoint'):
          os.mkdir('checkpoint')
  save_point = os.path.join("checkpoint", )
  torch.save(model.state_dict(), save_point + '.pt')
  print("Model saved!")

