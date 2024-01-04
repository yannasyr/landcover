## landcover
Il reste a faire :

-Data_augment + créer un ensemble de test commun pour tout le monde (yann)

-trouver un modele en plus de segfomer et Unet

-Rapport (overleaf ??)

-Diapo (baba)

## exemple de fonctionnement :

Cloner le repo sur sa machine

executer : python main.py --segformer --batch 16 --classes_to_ignore 0 1 

Ici on lance le main avec le modele segformer la taille de batch est 16 et on ignore les classes 0 et 1 qui sont no_data et clouds