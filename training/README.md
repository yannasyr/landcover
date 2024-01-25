# landcover
Il reste a faire :
- Rapport (en cours).
- Diaporama.
- Modèle avec augmentation et weighted loss. 

# exemple de fonctionnement :

Cloner le repo sur sa machine

executer : python main.py --segformer --batch 16 --classes_to_ignore 0 1 

Here we launch the segformer model with batch size 16 and we ignore classes 0 and 1 which are no_data and clouds 

You can look to file arg_parser to see all available arguments



