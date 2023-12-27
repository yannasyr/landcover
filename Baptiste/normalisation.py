import os
import rasterio
import numpy as np

def calculate_mean_and_std(image_path):
    with rasterio.open(image_path) as dataset:
        # Lire les canaux de l'image (les bandes)
        red = dataset.read(1)
        green = dataset.read(2)
        blue = dataset.read(3)
        nir = dataset.read(4)

        # Calculer la moyenne des canaux en divisant par 65535
        mean_red = np.mean(red)
        mean_green = np.mean(green)
        mean_blue = np.mean(blue)
        mean_nir = np.mean(nir)

        # Calculer l'écart type des canaux en divisant par 65535
        std_red = np.std(red)
        std_green = np.std(green)
        std_blue = np.std(blue)
        std_nir = np.std(nir)

    return (mean_red, mean_green, mean_blue, mean_nir), (std_red, std_green, std_blue, std_nir)

# Chemin du dossier contenant les images
folder_path = '/content/drive/MyDrive/small_dataset/images'

# Initialiser des listes pour stocker les moyennes et écarts types de toutes les images
all_means = []
all_stds = []

# Parcourir les fichiers du dossier
for filename in os.listdir(folder_path):
    if filename.endswith('.tif'):
        image_path = os.path.join(folder_path, filename)
        mean, std = calculate_mean_and_std(image_path)
        all_means.append(mean)
        all_stds.append(std)

# Calculer la moyenne et l'écart type global de toutes les images
global_mean = np.mean(all_means, axis=0)
global_std = np.mean(all_stds, axis=0)

print("Moyennes globales des canaux:", global_mean)
print("Écart types globaux des canaux:", global_std)