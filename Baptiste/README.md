# landcover
Il reste a faire :

-Data_augment (yann) voir albumentation

-Rapport (overleaf ??)

-Diapo (baba)

# exemple de fonctionnement :

Cloner le repo sur sa machine

executer : python main.py --segformer --batch 16 --classes_to_ignore 0 1 

Ici on lance le main avec le modele segformer la taille de batch est 16 et on ignore les classes 0 et 1 qui sont no_data et clouds


# Résultats :

Je peux mettre les modeles sur Drive(trop gros pour git) pour faire vos tests 

Les tests sont effectués sur le test set 

## SegFormer mit-B3 Sans les classes 0 1 8 :

KL = 0.009414138180034

Mean_iou: 0.6785667838462393

Mean accuracy: 0.7949927956010486

IoU per category [       nan        nan 0.67038836 0.75204877 0.77864742 0.66767148   0.65640219 0.40321102        nan 0.82159825]

OA 0.8362874562356236

## SegFormer mit-B5 Sans les classes 0 1 8 :

Entraînement terminé en 4h 3m 51s

KL = 0.008326901338153617

Mean_iou: 0.681108580508826

Mean accuracy: 0.7936043157681089

IoU per category [       nan        nan 0.67543331 0.75328871 0.78187457 0.67149005
 0.66350289 0.39616981        nan 0.82600073]
 
OA 0.8386775801021707

per category acc [       nan        nan 0.81124534 0.85776906 0.88339365 0.77037402
 0.80453262 0.51525236        nan 0.91266316]


## Unet Sans les classes 0 1 8 :

KL=0.02696049159138216

Mean_iou: 0.6582043868495238

Mean accuracy: 0.7559486492552072

IoU per category [       nan        nan 0.72456639 0.81718307 0.8170067  0.73129368   0.73694275 0.     nan 0.78043813]

OA 0.8743189385431362

## SegFormer mit-B5 RGb pré-entrainé Sans les classes 0 1 8 :

KL=0.0034651387263628015

Mean_iou: 0.7186073201358754

Mean accuracy: 0.8214072654344239

IoU per category [       nan        nan 0.7005971  0.80281907 0.78456663 0.65658247
 0.72356383 0.50915713        nan 0.85296501]

OA 0.8624028787160954

per category acc [       nan        nan 0.82535728 0.89151645 0.88751144 0.75894711
 0.84358665 0.62744141        nan 0.91549051]

 
## Deeplab : 
0.010837665145026999

Mean_iou: 0.6924559939043766

Mean accuracy: 0.8064495752718297

IoU per category [       nan        nan 0.64805616 0.77961633 0.76605886 0.68783081
 0.68169206 0.45407907        nan 0.82985867]

OA 0.845682461164968

per category acc [       nan        nan 0.7981537  0.87599234 0.87260965 0.79211492
 0.81325189 0.58943509        nan 0.90358944]
