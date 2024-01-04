import argparse



def parser():
    parser = argparse.ArgumentParser(description='Training')
    
    # Ajoutez des arguments pour spécifier le modèle
    parser.add_argument('--segformer',  action='store_true', help='SegFormer for semantic segmantation')
    parser.add_argument('--unet', action='store_true', help='Unet')
    parser.add_argument('--maskRcnn', action='store_true', help='mask2former')

    parser.add_argument('--batch_size', '-batch', default=16, type=int, help='batch_size')    
    
    # Analyser les arguments de la ligne de commande
    args = parser.parse_args()

    return args