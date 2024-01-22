from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from tifffile import TiffFile
import os


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
        print("idx = ", idx)
        image = self.train_data[idx]

        # Normalisez les valeurs des pixels dans la plage [0, 1]
        image = image.astype("float32")
        image = self.transform(image=image)['image']
    
        return image
    