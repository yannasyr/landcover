import argparse



def parser():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument("--test", action='store_true', help='start testing')
    parser.add_argument("--train", action='store_true', help='start training')
    parser.add_argument("--augmentation", action='store_true', help='training with augmentation on some classes')
    parser.add_argument('--segformer',  action='store_true', help='SegFormer for semantic segmantation')
    parser.add_argument('--unet', action='store_true', help='Unet')
    parser.add_argument('--classes_to_ignore','-classes_ign', default=[0, 1], nargs='+', type=int, help='List of numbers')
    parser.add_argument('--batch_size', '-batch', default=16, type=int, help='batch_size')    
    parser.add_argument('--save_model','-save', action='store_true', help='Saving checkpoints for current model')
    parser.add_argument('--mit_b3' , action='store_true', help='SegFormer Mit-B3') 
    parser.add_argument('--mit_b5' , action='store_true', help='SegFormer Mit-B5') 
    parser.add_argument('--num_channels','-chan', default=4, type=int, help='Number of spectral bands')
    parser.add_argument('--deeplab', action='store_true', help='deeplab model')
    parser.add_argument('--weighted', action='store_true', help='appli weighted losss')



    # Analyser les arguments de la ligne de commande
    args = parser.parse_args()

    return args