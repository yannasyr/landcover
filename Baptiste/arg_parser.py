import argparse



def parser():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--segformer',  action='store_true', help='SegFormer for semantic segmantation')
    parser.add_argument('--unet', action='store_true', help='Unet')
    parser.add_argument('--classes_to_ignore','-classes_ign', default=[0, 1], nargs='+', type=int, help='List of numbers')
    parser.add_argument('--batch_size', '-batch', default=16, type=int, help='batch_size')    
    
    # Analyser les arguments de la ligne de commande
    args = parser.parse_args()

    return args