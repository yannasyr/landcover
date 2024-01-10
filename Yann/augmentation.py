import cv2
import numpy as np
from tifffile import TiffFile, TiffWriter
import shutil
import random
import os

# -------------- FONCTIONS UTILES -----------------

def get_Y(mask2d):
  """
  Estimation du vecteur contenant les proportions de chaque classe dans une image segmentée:
  Y = [no_data, clouds, artificial, cultivated, broadleaf, coniferous, herbaceous, natural, snow, water]
  """
  occurrences = np.bincount(mask2d.flatten(), minlength=10)
  Y = occurrences / np.sum(occurrences)
  return Y

import cv2

def enhanceimage(image_path, mask_path):
    """ Actuellement, 6 nouvelles images. 
    - Changement de luminosité random deux fois.
    - rotation 90 degrés / -90 degrés.
    - flip vertical / horizontal.
    """

    enhanced_images = []
    enhanced_masks = []

    with TiffFile(image_path) as img:
        image = img.asarray()

    with TiffFile(mask_path) as msk:
        mask = msk.asarray()

    #Randomly adjust brightness
    value = random.randint(-3000, 3000)
    brightness_image = adjust_brightness(image, value)
    enhanced_images.append(brightness_image)
    enhanced_masks.append(mask)

    #Randomly adjust brightness
    value = random.randint(-3000, 3000)
    brightness_image = adjust_brightness(image, value)
    enhanced_images.append(brightness_image)
    enhanced_masks.append(mask)

    #Rotate the image by 90 degrees
    rotate_image = rotate_image_90(image, sens=1)
    rotate_mask = np.rot90(mask, k=1)
    enhanced_images.append(rotate_image)
    enhanced_masks.append(rotate_mask)

    #Rotate the image by -90 degrees
    rotate_image = rotate_image_90(image, sens=-1)
    rotate_mask = np.rot90(mask, k=-1)
    enhanced_images.append(rotate_image)
    enhanced_masks.append(rotate_mask)

    # Flip the image horizontally
    flip_image = flip_image_function(image, sens=-1)
    flip_mask = cv2.flip(mask,-1)
    enhanced_images.append(flip_image)
    enhanced_masks.append(flip_mask)

    # Flip the image vertically
    flip_image = flip_image_function(image, sens=1)
    flip_mask = cv2.flip(mask,1)
    enhanced_images.append(flip_image)
    enhanced_masks.append(flip_mask)
    
    return enhanced_images, enhanced_masks


def adjust_brightness(image, value):
    brightness_images = []
    for channel in range(image.shape[-1]):
        brightness_channel = np.int32(image[:, :, channel])
        brightness_channel = brightness_channel + value
        brightness_channel = np.clip(brightness_channel, 0, 65535)
        brightness_channel = np.uint16(brightness_channel)
        brightness_images.append(brightness_channel)

    return np.stack(brightness_images, axis=-1).astype(np.uint16) 

def rotate_image_90(image, sens=1):
    rotate_images = []
    for channel in range(image.shape[-1]):
        rotate_channel = np.rot90(image[:, :, channel], k=sens)
        rotate_images.append(rotate_channel)

    rotated_image = np.stack(rotate_images, axis=-1).astype(np.uint16)
    return rotated_image

def flip_image_function(image, sens=1):
    flip_images = []
    for channel in range(image.shape[-1]):
        flip_channel = cv2.flip(image[:, :, channel], sens)
        flip_images.append(flip_channel)

    return np.stack(flip_images, axis=-1).astype(np.uint16)

# ---------------------------- CODE APPLICATION ----------------------------

# Dossier contenant les images et les masques de segmentation
dossier_train = 'D:/my_git/landscape_data/dataset/train/'
dossier_images = dossier_train + 'images/'
dossier_masks = dossier_train + 'masks/'

# On veut sélectionner des images avec un taux de pixels important pour les classes à booster
# (water, city, natural, coniferous)
seuil_per_water = 0.3
seuil_per_city = 0.3
seuil_per_natural = 0.3
seuil_per_conif = 0.6

liste_images_interet = []

### Etape 1 : Identification des images intéressantes.
for mask_filename in os.listdir(dossier_masks):
    chemin_mask = os.path.join(dossier_masks, mask_filename)
    chemin_image = os.path.join(dossier_images, mask_filename)

    with TiffFile(chemin_mask) as tifm:
        mask = tifm.asarray()
        
    Y = get_Y(mask) # Y[2] = 'artificial' ; Y[5] = 'coniferous' ; Y[7] = 'natural' ; Y[9] = 'water'

    if Y[2] > seuil_per_city or Y[9] > seuil_per_water or Y[5] > seuil_per_conif  or Y[7] > seuil_per_natural :
        liste_images_interet.append(mask_filename)

print(" Images à augmenter identifiées")

dossier_augmented = 'D:/my_git/landscape_data/dataset/train_augmented/'
dossier_augmented_images = dossier_augmented + 'images/'
dossier_augmented_masks = dossier_augmented + 'masks/'

# Si le dossier "augmented" n'existe pas, on le créé. 
if not os.path.exists(dossier_augmented):
    os.makedirs(dossier_augmented)
    print(f"Le dossier {dossier_augmented} a été créé avec succès.")

if not os.path.exists(dossier_augmented_images):
    os.makedirs(dossier_augmented_images)  # Créer le dossier de manière récursive
    print(f"Le dossier {dossier_augmented_images} a été créé avec succès.")

if not os.path.exists(dossier_augmented_masks):
    os.makedirs(dossier_augmented_masks)  # Créer le dossier de manière récursive
    print(f"Le dossier {dossier_augmented_masks} a été créé avec succès.")

### Etape 2 : Augmentation des images intéressantes.
for image in liste_images_interet:
    image_path = os.path.join(dossier_images, image)
    mask_path = os.path.join(dossier_masks, image)
    enhanced_images, enhanced_masks = enhanceimage(image_path, mask_path) # image subit trois transformations -> 3 nouvelles images.

    for i, enhanced_image in enumerate(enhanced_images):
        transformed_image_path = os.path.join(dossier_augmented_images, image.replace('.tif', f'_t{i}.tif'))
        transformed_mask_path = os.path.join(dossier_augmented_masks, image.replace('.tif', f'_t{i}.tif'))
        with TiffWriter(transformed_image_path) as tifi:
            tifi.write(enhanced_image)
        with TiffWriter(transformed_mask_path) as tifm:
            tifm.write(enhanced_masks[i])

print("Images augmentées articiellement")

### Etape 3 : Déplacement des images restantes qui n'ont pas servit à l'augmentation.
for image_filename in os.listdir(dossier_images):
    src_image_path = os.path.join(dossier_images, image_filename)
    dest_image_path = os.path.join(dossier_augmented_images, image_filename)
    shutil.copy(src_image_path, dest_image_path)

for mask_filename in os.listdir(dossier_masks):
    src_mask_path = os.path.join(dossier_masks, mask_filename)
    dest_mask_path = os.path.join(dossier_augmented_masks, mask_filename)
    shutil.copy(src_mask_path, dest_mask_path)

print("Toutes les images et masques restants ont été copiés avec succès.")