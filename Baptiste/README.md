## landcover
Il reste a faire :

-Data_augment (yann) voir albumentation

-trouver un modele en plus de segfomer et Unet

-Rapport (overleaf ??)

-Diapo (baba)

-essayer RGB pre-entrainé

## exemple de fonctionnement :

Cloner le repo sur sa machine

executer : python main.py --segformer --batch 16 --classes_to_ignore 0 1 

Ici on lance le main avec le modele segformer la taille de batch est 16 et on ignore les classes 0 et 1 qui sont no_data et clouds


## Résultats :

Je peux mettre les modeles et le dossier de test que j'ai crée sur Drive(trop gros pour git) pour faire vos tests a noter que la KL divergence ne marche pas pour Unet 

Les tests sont effectué sur le test set 

KL = 0.009414138180034

SegFormer mit-B3 Sans les classes 0 1 8 :

Mean_iou: 0.6785667838462393

Mean accuracy: 0.7949927956010486

IoU per category [       nan        nan 0.67038836 0.75204877 0.77864742 0.66767148   0.65640219 0.40321102        nan 0.82159825]

OA 0.8362874562356236