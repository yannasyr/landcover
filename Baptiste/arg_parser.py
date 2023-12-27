import argparse



def parser():
    parser = argparse.ArgumentParser(description='Training')
    
    # Ajoutez des arguments pour spécifier le modèle
    parser.add_argument('--segformer',  action='store_true', help='SegFormer for semantic segmantation')
    parser.add_argument('--unet', action='store_true', help='Unet')

    
    # Analyser les arguments de la ligne de commande
    args = parser.parse_args()

    return args