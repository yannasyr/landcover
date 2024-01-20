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
from arg_parser import parser
from metrics import get_Y


args = parser()


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
        image = tifi.asarray()[:, :, :args.num_channels] 
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
        seuil_per_water = 0.3
        seuil_per_city = 0.3
        seuil_per_natural = 0.3
        seuil_per_conif = 0.6

        Y = get_Y(mask) # Y[2] = 'artificial' ; Y[5] = 'coniferous' ; Y[7] = 'natural' ; Y[9] = 'water'

        if (Y[2] > seuil_per_city or Y[9] > seuil_per_water or Y[5] > seuil_per_conif  or Y[7] > seuil_per_natural) and (self.transform_augm!=None):
            # Augmentation de données
            augmented = self.transform_augm(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        else:
            # Pas d'augmentation, transformation simple sur l'image.
            image = self.transform(image=image)['image']

            if args.segformer or args.deeplab:
                mask = torch.tensor(mask, dtype=torch.int64) 
                mask = mask.squeeze()

            elif args.unet:
                mask = torch.tensor(mask, dtype=torch.int64)
        
        classes_to_ignore = args.classes_to_ignore  # Replace with actual class indices

        # Apply the mask of ignorance
        ignore_mask = torch.ones_like(mask)
        for class_idx in classes_to_ignore:
            ignore_mask[mask == class_idx] = 0

            # Apply the mask to the ground truth label
            mask = mask * ignore_mask

        return image, mask
