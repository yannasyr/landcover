import os
import rasterio
import numpy as np

def calculate_mean_and_std(image_path):
    with rasterio.open(image_path) as dataset:
        # Lire les canaux de l'image (les bandes)
        red = dataset.read(1).astype(np.float32)
        green = dataset.read(2).astype(np.float32)
        blue = dataset.read(3).astype(np.float32)
        nir = dataset.read(4).astype(np.float32)

        # Aplatir les canaux dans une seule liste
        all_values = np.concatenate([red.flatten(), green.flatten(), blue.flatten(), nir.flatten()])

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

        # Calculer la valeur maximale parmi toutes les valeurs des canaux
        max_value = np.max(all_values)

    return (mean_red, mean_green, mean_blue, mean_nir), (std_red, std_green, std_blue, std_nir), max_value

# Chemin du dossier contenant les images
folder_path = '..\\dataset\\train\\images'

# Initialiser des listes pour stocker les moyennes et écarts types de toutes les images
all_means = []
all_stds = []
all_max_values = []

# Parcourir les fichiers du dossier
for filename in os.listdir(folder_path):
    if filename.endswith('.tif'):
        image_path = os.path.join(folder_path, filename)
        mean, std, max_value = calculate_mean_and_std(image_path)
        all_means.append(mean)
        all_stds.append(std)
        all_max_values.append(max_value)

# Calculer la moyenne et l'écart type global de toutes les images
global_mean = np.mean(all_means, axis=0)
global_std = np.mean(all_stds, axis=0)
global_max_value = np.max(all_max_values)

print("Moyennes globales des canaux:", global_mean)
print("Écart types globaux des canaux:", global_std)
print("Valeur maximale globale:", global_max_value)
