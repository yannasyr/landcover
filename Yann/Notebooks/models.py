from transformers import SegformerConfig
from transformers import SegformerForSemanticSegmentation
import torch
import torch.nn as nn
import torchvision.models as models


def segformer_eval(path_dict_model, num_channels=4, mit_b2=False, mit_b3=False, mit_b5=False, device='cpu'):
    
    if mit_b3:
        #Mit-B3 :
        config = SegformerConfig(
            num_labels=10,
            num_channels=4,
            semantic_loss_ignore_index=0,
            patch_sizes = [3, 2, 2, 2],
            depths=[3, 4, 18, 3],
            hidden_sizes=[64, 128, 320, 512],
            decoder_hidden_size=768,
        )
        model_name ="SegformerMit-B3"

    elif mit_b5:
        #Mit-B5 :
        config = SegformerConfig(
            num_labels=10,
            num_channels=4,
            semantic_loss_ignore_index=0,
            patch_sizes = [3, 2, 2, 2],
            depths=[3, 6, 40, 3],
            hidden_sizes=[64, 128, 320, 512],
            decoder_hidden_size=768,
        )
        model_name ="SegformerMit-B5"
        
    elif mit_b2:  
        #Mit-B2 :
        config = SegformerConfig(
            num_labels=10,
            num_channels=4,
            semantic_loss_ignore_index=0,
            patch_sizes = [3, 2, 2, 2],
            depths=[3, 4, 6, 3],
            hidden_sizes=[64, 128, 320, 512],
            decoder_hidden_size=768,
        )
        model_name ="SegformerMit-B2"

    # model = SegformerForSemanticSegmentation(config)
    # pretrained_state_dict = torch.load(path_dict_model, map_location=device)
    # model.load_state_dict(pretrained_state_dict)

    elif num_channels==3:
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5",
                                                         num_labels=10,
                                                         semantic_loss_ignore_index=0,
                                                         depths=[3, 6, 40, 3],
                                                         hidden_sizes=[64, 128, 320, 512],
                                                         decoder_hidden_size=768, 
                                                         )  
        model_name = 'mit-b5-RGB'
        pretrained_dict = torch.load(path_dict_model, map_location=device)
        model.load_state_dict(pretrained_dict)

    return model, model_name



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits = self.softmax(logits)
        return logits



class DeepLab4Channel(nn.Module):
    def __init__(self, num_classes):
        super(DeepLab4Channel, self).__init__()
        # Utilisez le modèle de base DeepLabV3 et ajustez la première couche pour accepter 4 canaux
        self.deepLabBase = models.segmentation.deeplabv3_resnet50(weights=None)
        # Remplacez la première couche pour accepter 4 canaux au lieu de 3
        self.deepLabBase.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Modifiez la dernière couche pour avoir le nombre de classes approprié
        self.deepLabBase.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.deepLabBase(x)