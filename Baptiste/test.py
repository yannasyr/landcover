from Landscapedata import LandscapeData
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from metrics import mesure_on_dataloader , affichage , compute_average_metrics
from arg_parser import parser
from transformers import SegformerForSemanticSegmentation
import torch


##Normalisation 
means =  [ 418.19976217,  703.34810956,  663.22678147, 3253.46844222]
stds =  [294.73191962, 351.31328415, 484.47475774, 793.73928079]

##transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
    ])
}



model = SegformerForSemanticSegmentation.from_pretrained("checkpoint\SegformerMit-B4_epoch0.pt",
                                                        num_labels=150, 
                                                        num_labels=10,
                                                        num_channels=4,
                                                        semantic_loss_ignore_index=0,
                                                        patch_sizes = [3, 2, 2, 2],
                                                        depths=[3, 4, 18, 3],
                                                        hidden_sizes=[64, 128, 320, 512],
                                                        decoder_hidden_size=768,                                                                                         
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
##dataset and dataloader
data_folder = "datasetV2\main\test"
test_dataset = LandscapeData(data_folder, transform=data_transforms['train']) 
print(len(test_dataset))
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


model.eval()
##Eval on val loader 
print(mesure_on_dataloader(test_loader,device,model,batch_size=8))
mean_iou, mean_accuracy, per_category_iou, Overall_acc = compute_average_metrics(model, test_loader,classes_to_ignore=[0,1,8]) #-> ignore les classes no data, clouds et snow -> aucune données
print("Mean_iou:", mean_iou)
print("Mean accuracy:", mean_accuracy)
print("IoU per category", per_category_iou)
print("OA", Overall_acc)
affichage(model,test_loader,device)




