import os
import random
from shutil import move
from torch.utils.data import DataLoader
from Landscapedata import LandscapeData

# Supposons que vous avez une classe LandscapeData définie qui charge les images et les masques
# data_transforms est votre transformation d'image

data_folder = "dataset - Copie\\train"
output_folder = "test"
os.makedirs(output_folder, exist_ok=True)

dataset = LandscapeData(data_folder)

# Créer un DataLoader pour faciliter l'itération sur l'ensemble de données
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Nombre d'images à extraire pour le fichier test
num_images_to_extract = 400

# Compteur pour suivre le nombre d'images extraites
count = 0

for idx, (image, mask) in enumerate(data_loader):
    # Vérifier si nous avons extrait le nombre souhaité d'images
    if count == num_images_to_extract:
        break

    # Générer un nom de fichier unique pour l'image et le masque
    filename = f"{idx}_{random.randint(1, 100000)}"

    # Déplacer l'image vers le dossier de test
    image_path = os.path.join(output_folder, f"image_{filename}.tif")
    move(image_path, output_folder)

    # Déplacer le masque correspondant vers le dossier de test
    mask_path = os.path.join(output_folder, f"mask_{filename}.tif")
    move(mask_path, output_folder)

    # Incrémenter le compteur
    count += 1

print(f"Extraction terminée. {count} images déplacées vers le dossier de test.")
