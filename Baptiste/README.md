## landcover
Il reste a faire :

-Data_augment + créer un ensemble de test commun pour tout le monde (yann) voir albumentation

-trouver un modele en plus de segfomer et Unet

-Rapport (overleaf ??)

-Diapo (baba)

## exemple de fonctionnement :

Cloner le repo sur sa machine

executer : python main.py --segformer --batch 16 --classes_to_ignore 0 1 

Ici on lance le main avec le modele segformer la taille de batch est 16 et on ignore les classes 0 et 1 qui sont no_data et clouds


## Résultats :
SegFormer sans les classes  [0,1,7,8,9]

Mean_iou: 0.6840779112434194

Mean accuracy: 0.8059646747377464

IoU per category [       nan        nan 0.64163087 0.73150159 0.75959704 0.65111578 0.63654428        nan        nan        nan]