from transformers import SegformerModel, SegformerConfig
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import torch
import torch.nn as nn

def segformer(lr=0.0001):

    config = SegformerConfig(
        num_labels=10,
        num_channels=4,
        patch_sizes = [3, 2, 2, 2]
    )

    model = SegformerForSemanticSegmentation(config)

    # define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    # move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    return model,optimizer


#A definir les autres modeles 