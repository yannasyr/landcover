import os
import shutil
from tifffile import TiffFile
from sklearn.model_selection import train_test_split

def create_train_test_folders(data_folder, train_ratio=0.90):
    image_folder = os.path.join(data_folder, 'images')
    all_files = os.listdir(image_folder)
    train_files, test_files = train_test_split(all_files, test_size=1 - train_ratio, random_state=42)


    train_folder = os.path.join(data_folder, 'train')
    test_folder = os.path.join(data_folder, 'test')

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    for filename in train_files:
        image_path = os.path.join(image_folder, filename)
        mask_path = image_path.replace("images", "masks")
        shutil.move(image_path, os.path.join(train_folder, 'images', filename))
        shutil.move(mask_path, os.path.join(train_folder, 'masks', filename.replace(".tif", ".tif")))

    for filename in test_files:
        image_path = os.path.join(image_folder, filename)
        mask_path = image_path.replace("images", "masks")
        shutil.move(image_path, os.path.join(test_folder, 'images', filename))
        shutil.move(mask_path, os.path.join(test_folder, 'masks', filename.replace(".tif", ".tif")))

# Example usage:
data_folder = "datasetV2\\main"  # Replace with your actual dataset path
create_train_test_folders(data_folder)
