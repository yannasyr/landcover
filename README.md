# ENS Challenge Data 2021 : Land cover predictive modeling from satellite images

Authors : [yannasyr](https://github.com/yannasyr) / [BaptistePodvin](https://github.com/BaptistePodvin) / [Nom Auteur 1](https://github.com/lenadhb)
This repository and project is motivated by a machine-learning workshop as part of the ISSD-3A formation at Télécom Physique Strasbourg. 

The idea was to participate for the Challenge Data competition [“Land cover predictive modeling from satellite images”](https://challengedata.ens.fr/challenges/48) provided by Preligens.

You can download the data as an archive containing the training images and masks, as well as the test images, from the challenge page.

The dataset folder should be like this :
```
dataset_UNZIPPED
├── test
│   └── images
│       ├── 10087.tif
│       ├── 10088.tif
│       ├── 10089.tif
│       ├── 10090.tif
        ... (5043 files)
└── train
    ├── images
    │   ├── 10000.tif
    │   ├── 10001.tif
    │   ├── 10002.tif
    │   ├── 10003.tif
        ... (18491 files)
    └── masks
        ├── 10000.tif
        ├── 10001.tif
        ├── 10002.tif
        ├── 10003.tif
        ... (18491 files)
```
The images are 16-bits GeoTIFF files of size (256,256,4) and the masks are 8-bits GeoTIFF files of size (256,256).

## Python environment

The `requirements.txt` file contains all the necessary libraries. To install, you can use `pip install -r requirements.txt`, or any other way you know. 

## Repo organisation

`archives/` mainly contains notebooks that were written at the beginning of the project. To have some visualization on the classes, the images and the masks, we recommend to scroll through `yannv3.ipynb`

`csv_output/` contains scripts to produce the .csv file expected for the challenge submission (cf. the challenge data link.)

`training/` is the directory where the **interesting** part is at. It contains a README.md to explain how to use it. 
