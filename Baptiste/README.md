## landcover
Il reste a faire :

-Data_augment (yann) voir albumentation

-Faire un test Set pour nos metriques Urgent pour lancer les entrainements. 

-trouver un modele en plus de segfomer et Unet

-Rapport (overleaf ??)

-Diapo (baba)

-essayer RGB pre-entrainé

## exemple de fonctionnement :

Cloner le repo sur sa machine

executer : python main.py --segformer --batch 16 --classes_to_ignore 0 1 

Ici on lance le main avec le modele segformer la taille de batch est 16 et on ignore les classes 0 et 1 qui sont no_data et clouds


## Résultats :

Je peux mettre les modeles sur Drive(trop gros pour git) pour faire vos tests a noter que la KL divergence ne marche pas pour Unet 

SegFormer miT-B0 sans les classes  [0,1,7,8,9]

Mean_iou: 0.6840779112434194

Mean accuracy: 0.8059646747377464

IoU per category [       nan        nan 0.64163087 0.73150159 0.75959704 0.65111578 0.63654428        nan        nan        nan]



Unet  sans les classes  [0,1,7,8,9]

Mean_iou: 0.7583717354907262

Mean accuracy: 0.855796475454197

IoU per category [       nan        nan 0.73023121 0.80774726 0.8048192  0.72396396  0.72509706        nan        nan        nan]

OA 0.8712198427578629