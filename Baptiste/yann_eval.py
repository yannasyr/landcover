import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.transforms import ToTensor
from transformers import  SegformerForSemanticSegmentation

import albumentations as A
from albumentations.pytorch import ToTensorV2

import os 
import csv
import numpy as np
from tifffile import TiffFile
from tqdm import tqdm

# ------------ MODEL

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits = self.softmax(logits)
        return logits

# ----------- 2ème modèle.

def segformer(num_channels=3, test=True):

    if num_channels==3:
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5",
                                                            num_labels=10,
                                                            semantic_loss_ignore_index=0,
                                                            depths=[3, 6, 40, 3],
                                                            hidden_sizes=[64, 128, 320, 512],
                                                            decoder_hidden_size=768, 
                                                            )  
        if test :
            pretrained_dict = torch.load("D:/my_git/TPE-3A/SegformerMit-RGB_epoch35.pt", map_location=torch.device('cpu'))
            model.load_state_dict(pretrained_dict)

    return model

# -------------- DATASET 
    
def numpy_parse_image(image_path):
    with TiffFile(image_path) as tifi:
        image = tifi.asarray()[:,:,:3]
    return image

class LandscapeData_eval(Dataset):

    def __init__(self, data_folder, transform=ToTensor()):
        self.data_folder = data_folder
        self.transform = transform

        # Liste des noms de fichiers dans les dossiers
        image_files = os.listdir(os.path.join(data_folder, 'images'))
        print("Taille image_files = ", len(image_files))

        # Utilisez numpy_parse_image_mask pour charger les images et les masques
        self.train_data = [numpy_parse_image(os.path.join(data_folder, 'images', filename)) for filename in image_files]

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        image = self.train_data[idx]

        # Normalisez les valeurs des pixels dans la plage [0, 1]
        image = image.astype("float32")
        image = self.transform(image=image)['image']
    
        return image

# ---------------- EVAL
    
def get_Y(mask2d):
    occurrences = np.bincount(mask2d.flatten(), minlength=10)
    Y = occurrences / np.sum(occurrences)
    return Y
        
def evaluate(model, dataloader, device, output_csv_path, index):
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs in tqdm(dataloader):
            inputs = inputs.to(device)

            # Obtenez les prédictions du modèle
            outputs = model(inputs)
            # Pour le segformer
            outputs = outputs.logits

            # Obtenez les classes prédites en utilisant la classe avec la probabilité la plus élevée
            _, predicted_classes = outputs.max(dim=1)

            # Convertir les masques en numpy array
            predicted_masks_np = predicted_classes.cpu().numpy()

            # Appliquer la fonction get_Y à chaque masque
            batch_predictions = [get_Y(mask2d) for mask2d in predicted_masks_np]
            predictions.extend(batch_predictions)

    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Écrire l'en-tête du fichier CSV
        csv_writer.writerow(['sample_id'] + ['no_data'] + ['clouds'] + ['artificial'] + ['cultivated'] +
                            ['broadleaf'] + ['coniferous'] + ['herbaceous'] + ['natural'] + ['snow'] + ['water'])

        # Écrire les résultats pour chaque image
        for i, prediction in enumerate(predictions):
            # Formatage de chaque valeur de prédiction avec un pourcentage
            formatted_predictions = [f'{pourcent:.6f}' for pourcent in prediction]
            
            # Écrire la ligne dans le fichier CSV
            csv_writer.writerow([index[i]] + formatted_predictions)

    return predictions

# -------------- MAIN CODE 

means =  (418.19976217, 703.34810956, 663.22678147, 3253.46844222)
stds =  (294.73191962, 351.31328415, 484.47475774, 793.73928079)

data_transforms = {
    'test': A.Compose([
        A.Normalize(mean=means, std=stds, max_pixel_value=1.0),
        ToTensorV2()
    ])
}

# model = UNet2(4, 10, bilinear=False)
# pretrained_state_dict = torch.load("D:/my_git/TPE-3A/Unet_epoch85.pt", map_location=torch.device('cpu'))
# model.load_state_dict(pretrained_state_dict)
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
# model_name ="Unet"

model = segformer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_folder = 'D:/my_git/landscape_data/dataset/test/'

index_photos = os.listdir(data_folder+'images/')
index = [name.replace('.tif', '') for name in index_photos]
eval_dataset = LandscapeData_eval(data_folder, transform=data_transforms['test'])  

print(eval_dataset.__len__())
eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False)
predictions = evaluate(model, eval_loader, device, 'C:/Users/kille/results_segformerRGB.csv', index=index)

print(len(predictions))