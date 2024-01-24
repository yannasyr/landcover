# landcover
Il reste a faire :
- Rapport (en cours).
- Diaporama.
- Modèle avec augmentation et weighted loss. 

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

 
## Deeplab Sans les classes 0 1 8 :: 
0.010837665145026999

Mean_iou: 0.6924559939043766

Mean accuracy: 0.8064495752718297

IoU per category [       nan        nan 0.64805616 0.77961633 0.76605886 0.68783081
 0.68169206 0.45407907        nan 0.82985867]

OA 0.845682461164968

per category acc [       nan        nan 0.7981537  0.87599234 0.87260965 0.79211492
 0.81325189 0.58943509        nan 0.90358944]

## Segformer mit-b5 augmentation Sans les classes 0 1 8 :
0.02909050583945016

Mean_iou: 0.627369287231402

Mean accuracy: 0.7654155547258374

IoU per category [       nan        nan 0.63001772 0.7163735  0.74471817 0.64981787
 0.60404711 0.32291055        nan 0.72370009]

OA 0.8072430155140854

per category acc [       nan        nan 0.78805123 0.84135008 0.84782107 0.80408215
 0.7437545  0.51697298        nan 0.81587687]

## U-Net Augmentation + Weighted Loss + Classes_Ignored = [0,1,8]

0.0433296225277715

Mean_iou: 0.6328018673377286

Mean accuracy: 0.8451287015716559

IoU per category [       nan        nan 0.59692087 0.72941721 0.76284218 0.66119263
 0.65187745 0.286752          nan 0.74061072]
 
OA 0.8177371849598891

per category acc [       nan        nan 0.89406548 0.79812128 0.84933898 0.87729572
 0.78020423 0.89702264        nan 0.81985258]

## U-Net Augmentation (p=0.2) + Weighted Loss(avec natural=1) + Classes_Ignored = [0,1,8]

 0.03454798191315818
 
Mean_iou: 0.6117048347202247

Mean accuracy: 0.7622194414943716

IoU per category [       nan        nan 0.61669964 0.77656405 0.77861903 0.67556434
 0.69882105 0.                nan 0.73566573]
 
OA 0.843869509329675

per category acc [       nan        nan 0.94684003 0.84664301 0.85880237 0.89257575
 0.80618614 0.                nan 0.98448879]

 ## Segformer MIT-B5 + Weighted Loss + Classes_Ignored = [0,1,8]

 0.07796208727084196
 
Mean_iou: 0.6140343481521935

Mean accuracy: 0.8314605111467921

IoU per category [       nan        nan 0.56356087 0.69113471 0.74299229 0.6405514
 0.61913755 0.26944834        nan 0.77141528]
 
OA 0.7969969071993864

per category acc [       nan        nan 0.88306001 0.76163612 0.83761083 0.86135998
 0.76266096 0.85404544        nan 0.85985024]
