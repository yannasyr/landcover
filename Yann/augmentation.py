import cv2
import numpy as np
from tifffile import TiffFile, TiffWriter
import shutil
import random
import os

# /!\ PAS ENCORE AU POINT /!\
# /!\ PAS ENCORE AU POINT /!\
# /!\ PAS ENCORE AU POINT /!\
# /!\ PAS ENCORE AU POINT /!\
# /!\ PAS ENCORE AU POINT /!\
# /!\ PAS ENCORE AU POINT /!\
# /!\ PAS ENCORE AU POINT /!\
# /!\ PAS ENCORE AU POINT /!\


# -------------- FONCTIONS UTILES -----------------

def get_Y(mask2d):
  """
  Estimation du vecteur contenant les proportions de chaque classe dans une image segmentée:
  Y = [no_data, clouds, artificial, cultivated, broadleaf, coniferous, herbaceous, natural, snow, water]
  """
  occurrences = np.bincount(mask2d.flatten(), minlength=10)
  Y = occurrences / np.sum(occurrences)
  return Y

def enhanceimage(image_path, mask_path):

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

    #Rotate the image by 90 degrees
    rotate_image = rotate_image_90(image)
    rotate_mask = np.rot90(mask)
    enhanced_images.append(rotate_image)
    enhanced_masks.append(rotate_mask)

    # Flip the image horizont:ally
    flip_image = flip_image_horizontally(image)
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

def rotate_image_90(image):
    rotate_images = []
    for channel in range(image.shape[-1]):
        rotate_channel = np.rot90(image[:, :, channel], k=1)  # Rotate by 90 degrees
        rotate_images.append(rotate_channel)

    rotated_image = np.stack(rotate_images, axis=-1).astype(np.uint16)
    return rotated_image

def flip_image_horizontally(image):
    flip_images = []
    for channel in range(image.shape[-1]):
        flip_channel = cv2.flip(image[:, :, channel], 1)
        flip_images.append(flip_channel)

    return np.stack(flip_images, axis=-1).astype(np.uint16)


# -------------- CODE APPLICATION -----------------


# Dossier contenant les masques de segmentation ()
dossier_train = 'D:/my_git/landscape_data/dataset/small_dataset_bis/'
dossier_augmented = 'D:/my_git/landscape_data/dataset/small_dataset_bis_augmented/'
dossier_images = dossier_train + 'images/'
dossier_masks = dossier_train + 'masks/'

# Si le dossier "augmented" n'existe pas, on le créé. 
if not os.path.exists(dossier_augmented):
    os.makedirs(dossier_augmented)
    print(f"Le dossier {dossier_augmented} a été créé avec succès.")

seuil_per_water = 0.3
seuil_per_city = 0.3

# Déplacement des images intéressantes.
for mask_filename in os.listdir(dossier_masks):
    chemin_mask = os.path.join(dossier_masks, mask_filename)
    chemin_image = os.path.join(dossier_images, mask_filename)

    with TiffFile(chemin_mask) as tifm:
        mask = tifm.asarray()
        
    Y = get_Y(mask) # Y[2] = 'artificial' and Y[9] = 'water'

    if Y[2] > seuil_per_city or Y[9] > seuil_per_water:
        destination_mask = os.path.join(dossier_augmented, 'masks', mask_filename)
        destination_image = os.path.join(dossier_augmented, 'images', mask_filename)

        os.makedirs(os.path.dirname(destination_mask), exist_ok=True)
        os.makedirs(os.path.dirname(destination_image), exist_ok=True)

        shutil.copy(chemin_mask, destination_mask)
        shutil.copy(chemin_image, destination_image)

dossier_augmented_image = dossier_augmented + 'images/'
dossier_augmented_masks = dossier_augmented + 'masks/'

for image in os.listdir(dossier_augmented_image):
    image_path = os.path.join(dossier_augmented_image, image)
    mask_path = os.path.join(dossier_augmented_masks, image)
    enhanced_images, enhanced_masks = enhanceimage(image_path, mask_path)

    for i, enhanced_image in enumerate(enhanced_images):
        transformed_image_path = os.path.join(dossier_augmented_image, image.replace('.tif', f'_t{i}.tif'))
        transformed_mask_path = os.path.join(dossier_augmented_masks, image.replace('.tif', f'_t{i}.tif'))
        with TiffWriter(transformed_image_path) as tifi:
            tifi.write(enhanced_image)
        with TiffWriter(transformed_mask_path) as tifm:
            tifm.write(enhanced_masks[i])