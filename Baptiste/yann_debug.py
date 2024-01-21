import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from sklearn.model_selection import train_test_split
from skimage.transform import resize
from collections import OrderedDict
from tifffile import TiffFile
from pathlib import Path
import os

from LandCoverData import LandCoverData
from metrics import get_Y

import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np

class LandCoverData():
    """Class to represent the S2GLC Land Cover Dataset for the challenge,
    with useful metadata and statistics.
    """
    # image size of the images and label masks
    IMG_SIZE = 256
    # the images are RGB+NIR (4 channels)
    N_CHANNELS = 4
    # we have 9 classes + a 'no_data' class for pixels with no labels (absent in the dataset)
    N_CLASSES = 10
    CLASSES = [
        'no_data',
        'clouds',
        'artificial',
        'cultivated',
        'broadleaf',
        'coniferous',
        'herbaceous',
        'natural',
        'snow',
        'water'
    ]
    # classes to ignore because they are not relevant. "no_data" refers to pixels without
    # a proper class, but it is absent in the dataset; "clouds" class is not relevant, it
    # is not a proper land cover type and images and masks do not exactly match in time.
    IGNORED_CLASSES_IDX = [0, 1]

    # The training dataset contains 18491 images and masks
    # The test dataset contains 5043 images and masks
    TRAINSET_SIZE = 18491
    TESTSET_SIZE = 5043

    # for visualization of the masks: classes indices and RGB colors
    CLASSES_COLORPALETTE = {
        0: [0,0,0],
        1: [255,25,236],
        2: [215,25,28],
        3: [211,154,92],
        4: [33,115,55],
        5: [21,75,35],
        6: [118,209,93],
        7: [130,130,130],
        8: [255,255,255],
        9: [43,61,255]
        }
    CLASSES_COLORPALETTE = {c: np.asarray(color) for (c, color) in CLASSES_COLORPALETTE.items()}

    # statistics
    # the pixel class counts in the training set
    TRAIN_CLASS_COUNTS = np.array(
        [0, 20643, 60971025, 404760981, 277012377, 96473046, 333407133, 9775295, 1071, 29404605]
    )
    # the minimum and maximum value of image pixels in the training set
    TRAIN_PIXELS_MIN = 1
    TRAIN_PIXELS_MAX = 24356



def numpy_parse_image_mask(image_path):
    """Load an image and its segmentation mask as numpy arrays and returning a tuple
    Args:
        image_path : path to image
    Returns:
        (numpy.array[uint16], numpy.array[uint8]): the image and mask arrays
    """
    # image_path = Path(image_path)
    # get mask path from image path:
    # image should be in a images/<image_id>.tif subfolder, while the mask is at masks/<image_id>.tif
    mask_path = image_path.replace("images","masks")
    with TiffFile(image_path) as tifi, TiffFile(mask_path) as tifm:
        image = tifi.asarray()[:, :, :4] 
        mask = tifm.asarray()
    return image, mask



class LandscapeData(Dataset):

    N_CHANNELS = LandCoverData.N_CHANNELS
    IMG_SIZE = LandCoverData.IMG_SIZE
    TRAIN_PIXELS_MAX = LandCoverData.TRAIN_PIXELS_MAX

    def __init__(self, data_folder, transform=ToTensor(), transform_augm=None):
        self.data_folder = data_folder
        self.transform = transform
        self.transform_augm = transform_augm

        # Liste des noms de fichiers dans les dossiers
        image_files = os.listdir(os.path.join(data_folder, 'images'))

        # Utilisez numpy_parse_image_mask pour charger les images et les masques
        self.train_data = [numpy_parse_image_mask(os.path.join(data_folder, 'images', filename)) for filename in image_files]

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        image, mask = self.train_data[idx]
        channels, height, width = self.N_CHANNELS, self.IMG_SIZE, self.IMG_SIZE

        # Normalisez les valeurs des pixels dans la plage [0, 1]
        image = image.astype("float32")
        mask = mask.astype("int64")

        # Memoire seuils : 0.3 / 0.3 / 0.3 / 0.6
        seuil_per_water = 0.00001
        seuil_per_city = 0.00001
        seuil_per_natural = 0.00001
        seuil_per_conif = 0.00001

        Y = get_Y(mask) # Y[2] = 'artificial' ; Y[5] = 'coniferous' ; Y[7] = 'natural' ; Y[9] = 'water'

        if (Y[2] > seuil_per_city or Y[9] > seuil_per_water or Y[5] > seuil_per_conif  or Y[7] > seuil_per_natural) and (self.transform_augm!=None):
            # Augmentation de données
            print("augmented!")
            augmented = self.transform_augm(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        else:
            # Pas d'augmentation, transformation simple sur l'image.
            image = self.transform(image=image)['image']

            if True or False:
                mask = torch.tensor(mask, dtype=torch.int64) 
                mask = mask.squeeze()

            elif False:
                mask = torch.tensor(mask, dtype=torch.int64)
        
        classes_to_ignore = [0,1,8]  # Replace with actual class indices

        # Apply the mask of ignorance
        ignore_mask = torch.ones_like(mask)
        for class_idx in classes_to_ignore:
            ignore_mask[mask == class_idx] = 0

            # Apply the mask to the ground truth label
            mask = mask * ignore_mask

        return image, mask
    

### ----------- PARTIE DEBUG DE YANN --------- ###

# Normalisation
if True :
    means =  (418.19976217,  703.34810956,  663.22678147, 3253.46844222)
    stds =  (294.73191962, 351.31328415, 484.47475774, 793.73928079)
else :
    means =  [ 418.19976217,  703.34810956,  663.22678147]
    stds =  [294.73191962, 351.31328415, 484.47475774]

# Transformations 
data_transforms = {
    'train': A.Compose([
         A.Normalize(mean=means, std=stds, max_pixel_value=1),
         ToTensorV2()
    ]),
    'train_augmentation': A.Compose([
        A.HorizontalFlip(p=0.7),
        A.VerticalFlip(p=0.7),
        A.RandomRotate90(p=0.7),
        A.Transpose(p=0.7),
        A.Normalize(means, stds),
        ToTensorV2()
    ]),
    'test': A.Compose([
         A.Normalize(mean=means, std=stds, max_pixel_value=1),
         ToTensorV2() 
    ])
}
    
# A.Compose([
#         A.Normalize(mean=means, std=stds),
#         ToTensorV2()
# ------------- DATASET & DATALOADER ----------- 

# Définir le chemin du dossier d'entraînement
data_folder = 'D:/my_git/landscape_data/dataset/small_dataset/'

# Créer un objet Dataset pour l'ensemble.
full_dataset = LandscapeData(data_folder, transform=None)  

# Division en ensemble train (80%)  / validation (10%) / test (10%)
train_size = int(0.8 * len(full_dataset))
val_size = int(0.10 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

train_dataset.dataset.transform = data_transforms['train']
val_dataset.dataset.transform = data_transforms['test']
test_dataset.dataset.transform = data_transforms['test']    

# A changer pour le debug
if True :
    train_dataset.dataset.transform_augm = data_transforms['train_augmentation']

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

images, masks = next(iter(train_loader))
im1, im2, im3, im4 = images
ms1, ms2, ms3, ms4 = masks 

images, masks = next(iter(val_loader))
im1, im2, im3, im4 = images
ms1, ms2, ms3, ms4 = masks 

images, masks = next(iter(test_loader))
im1, im2, im3, im4 = images
ms1, ms2, ms3, ms4 = masks 

