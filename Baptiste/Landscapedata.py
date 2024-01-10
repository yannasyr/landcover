from LandCoverData import LandCoverData
from tifffile import TiffFile
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from pathlib import Path
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
import os
import torch
from arg_parser import parser

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
        image = tifi.asarray()
        mask = tifm.asarray()
    return image, mask


class LandscapeData(Dataset):
    N_CHANNELS = LandCoverData.N_CHANNELS
    IMG_SIZE = LandCoverData.IMG_SIZE
    TRAIN_PIXELS_MAX = LandCoverData.TRAIN_PIXELS_MAX

    def __init__(self, data_folder, transform=ToTensor()):
        self.data_folder = data_folder
        self.transform = transform

        # Liste des noms de fichiers dans les dossiers
        image_files = os.listdir(os.path.join(data_folder, 'images'))

        # Séparation des données en ensembles d'entraînement, de validation et de test
        train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)
        train_files, val_files = train_test_split(train_files, test_size=0.1, random_state=42)

        # Utilisez numpy_parse_image_mask pour charger les images et les masques
        self.train_data = [numpy_parse_image_mask(os.path.join(data_folder, 'images', filename)) for filename in train_files]
        self.val_data = [numpy_parse_image_mask(os.path.join(data_folder, 'images', filename)) for filename in val_files]
        self.test_data = [numpy_parse_image_mask(os.path.join(data_folder, 'images', filename)) for filename in test_files]

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        image, label = self.train_data[idx]
        channels, height, width = self.N_CHANNELS, self.IMG_SIZE, self.IMG_SIZE

        # Normalisez les valeurs des pixels dans la plage [0, 1]
        image = image.astype("float32")
        label = label.astype("int64")

        image = self.transform(image)
        
        args = parser()
        classes_to_ignore = args.classes_to_ignore  # Replace with actual class indices

        if args.segformer : 
            # Modifiez la transformation pour le masque
            label = torch.tensor(label, dtype=torch.int64)  # Convertir en torch.Tensor
            label = label.squeeze()  # Supprimer la dimension ajoutée
        elif args.unet :
            label = torch.tensor(label, dtype=torch.int64)  # Convertir en torch.Tensor
        # Apply the mask of ignorance
        ignore_mask = torch.ones_like(label)
        for class_idx in classes_to_ignore:
            ignore_mask[label == class_idx] = 0

            # Apply the mask to the ground truth label
            label = label * ignore_mask


        return image, label
