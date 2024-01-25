import torch
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import os 
import csv
import numpy as np
from tqdm import tqdm

from models import UNet2
from models import segformer_eval
from Landscapedata import LandscapeData_eval

# ------------- FONCTIONS UTILES -----------

def get_Y(mask2d):
    occurrences = np.bincount(mask2d.flatten(), minlength=10)
    Y = occurrences / np.sum(occurrences)
    return Y

def evaluate(model, model_name, dataloader, device, output_csv_path, index):
    model = model.to(device)
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs in tqdm(dataloader):
            inputs = inputs.to(device)

            # Obtenez les prédictions du modèle
            outputs = model(inputs)
            # Pour le segformer
            if model_name == 'segformer':
                outputs = outputs.logits

            # Obtenez les classes prédites en utilisant la classe avec la probabilité la plus élevée
            _, predicted_classes = outputs.max(dim=1)

            # Convertir les masques en numpy array
            predicted_masks_np = predicted_classes.cpu().numpy()

            # Appliquer la fonction get_Y à chaque masque
            batch_predictions = [get_Y(mask2d) for mask2d in predicted_masks_np]
            predictions.extend(batch_predictions)

    for i, prediction in enumerate(predictions):
            # Formatage de chaque valeur de prédiction avec un pourcentage
            formatted_predictions = [f'{pourcent:.6f}' for pourcent in prediction]


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

# -------------- MAIN CODE -----------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- VARIABLES A MODIFIER ---------------------------------------------
eval_folder = 'D:/my_git/landscape_data/dataset/small_dataset/'
eval_csv_output_name = 'C:/Users/kille/results_small_segformer.csv'
path_model = "D:/my_git/TPE-3A/SegformerMit-B5_epoch35.pt"
choix_modele = 'segformer' # soit 'unet', soit 'segformer'. Si 'segformer', modifier bonnes options dans la fonction segformer()
nb_channels = 4
# ----------------------------------------------------------------------

# Choix du modèle. 
if choix_modele == 'unet':
    model = UNet2(4, 10, bilinear=False)
    pretrained_state_dict = torch.load(path_model, map_location=device)
    model.load_state_dict(pretrained_state_dict)
    model_name ="Unet"
else:
    model, model_name = segformer_eval(path_dict_model=path_model, num_channels=nb_channels, mit_b5=True, device=device)

# Transformées ----------------------------------------------------
means =  (418.19976217, 703.34810956, 663.22678147, 3253.46844222)
stds =  (294.73191962, 351.31328415, 484.47475774, 793.73928079)

data_transforms = {
    'test': A.Compose([
        A.Normalize(mean=means, std=stds, max_pixel_value=1.0),
        ToTensorV2()
    ])
}
# ----------------------------------------------------------------------

index_photos = os.listdir(eval_folder+'images/')
index = [name.replace('.tif', '') for name in index_photos]

eval_dataset = LandscapeData_eval(eval_folder, transform=data_transforms['test'], num_channels=nb_channels)  
eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
predictions = evaluate(model,choix_modele, eval_loader, device, eval_csv_output_name, index=index)

print("Nombre de prédictions : ", len(predictions))