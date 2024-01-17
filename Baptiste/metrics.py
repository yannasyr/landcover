import numpy as np
import torch
import matplotlib.pyplot as plt  # Fix import statement
from LandCoverData import LandCoverData
from datasets import load_metric
import torch.nn as nn
from arg_parser import parser
args = parser()



def get_Y(mask2d):
  occurrences = np.bincount(mask2d.flatten(), minlength=10)
  Y = occurrences / np.sum(occurrences)
  return Y

def klmetric(y_t, y_p, eps):
    sum = 0
    for i in range(len(y_t)):
        if y_t[i] != 0:  # Ignorer la classe avec le label 0
            sum += (y_t[i] + eps) * np.log((y_t[i] + eps) / (y_p[i] + eps))
    return sum


def mesure_on_batch(batch_gt, batch_predi, batch_size=args.batch_size):
  mean = 0
  for i in range(batch_size):
    Y_pred = get_Y(batch_gt[i].cpu().numpy())
    Y_truth = get_Y(batch_predi[i].cpu().numpy())
    mean += klmetric(Y_truth, Y_pred, 10e-8)
  return mean / batch_size

def mesure_on_dataloader(val_loader,device,model,batch_size=args.batch_size):

  mean = 0
  for i in range(len(val_loader)):
    test_inputs, test_targets = next(iter(val_loader))
    test_pixels_values = test_inputs.to(device)
    model.eval()
    with torch.no_grad():
      if args.segformer :
        test_outputs = model(pixel_values=test_pixels_values)
        test_logits = test_outputs.logits
      else :
         test_outputs=model(test_pixels_values)
         test_logits=test_outputs

    _, predicted_labels = torch.max(test_logits, dim=1)

    mean += mesure_on_batch(test_targets, predicted_labels,batch_size=args.batch_size)

  return mean / len(val_loader)


def affichage(model,data_loader,device):
  CLASSES_COLORPALETTE=LandCoverData.CLASSES_COLORPALETTE

  # Assuming val_loader is defined in your code
  val_inputs, val_targets = next(iter(data_loader))

  val_pixel_values = val_inputs.to(device)

  # Set the model to evaluation mode
  model.eval()
  # Perform inference on the validation image
  with torch.no_grad():
      if args.segformer :
        val_outputs = model(pixel_values=val_pixel_values)
        val_logits = val_outputs.logits
      if args.unet :
         val_outputs=model(val_pixel_values)
         val_logits=val_outputs

  # Convert logits to predicted labels
  _, predicted_labels = torch.max(val_logits, dim=1)

  # Visualize the results for one image
  for i in range(4):  # Assuming batch size is 4
      plt.figure(figsize=(15, 5))

      # Original image
      original_image = val_inputs[i].permute(1, 2, 0).cpu().numpy()
      original_image = original_image[:, :, :3] # Utilisez seulement les 3 premiers canaux et normalisez à [0, 1]
      original_image=original_image / original_image.max()
      plt.subplot(1, 3, 1)
      plt.imshow(original_image)
      plt.title("Original Image")

      # Ground truth mask
      plt.subplot(1, 3, 2)
      ground_truth_mask = val_targets[i].cpu().numpy()
      ground_truth_mask_rgb = np.zeros((ground_truth_mask.shape[0], ground_truth_mask.shape[1], 3), dtype=np.uint8)
      for cls, color in CLASSES_COLORPALETTE.items():
          ground_truth_mask_rgb[ground_truth_mask == cls] = color
      plt.imshow(ground_truth_mask_rgb)
      plt.title("Ground Truth Mask")

      # Predicted mask
      plt.subplot(1, 3, 3)
      # Rescale logits to original image size
      upsampled_logits = torch.nn.functional.interpolate(val_logits, size=val_pixel_values.shape[2:], mode='bicubic')
      # Apply argmax on the class dimension
      seg = upsampled_logits.argmax(dim=1)[i].cpu().numpy()
      color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
      for label, color in enumerate(CLASSES_COLORPALETTE.values()):
          color_seg[seg == label] = color


      # Show image + mask
      img = original_image  + color_seg
      img = img.astype(np.uint8)

      plt.imshow(img)
      plt.title("Predicted Mask")

      plt.show()
  


def compute_average_metrics(model, val_loader, classes_to_ignore=[]):
    model.eval()
    metric = load_metric("mean_iou",trust_remote_code=True)
    with torch.no_grad():
        for inputs, targets in val_loader:
            pixel_values = inputs.to('cuda:0')
            labels = targets.to('cuda:0')
            if args.unet : 
              outputs = model(pixel_values)
              logits=outputs
            else : 
              outputs = model(pixel_values=pixel_values, labels=labels)
              logits = outputs.logits

            upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            predicted = upsampled_logits.argmax(dim=1)

            # Créer un masque pour ignorer certaines classes
            ignore_mask = torch.ones_like(labels)
            for class_idx in classes_to_ignore:
                ignore_mask[labels == class_idx] = 0

            # Appliquer le masque d'ignorance aux prédictions et références
            predicted = predicted * ignore_mask
            labels = labels * ignore_mask

            metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

    metrics = metric.compute(num_labels=10,
                             ignore_index= 0,
                             reduce_labels=False  # we've already reduced the labels before
                             )

    mean_iou = metrics["mean_iou"]
    mean_accuracy = metrics["mean_accuracy"]
    per_category_iou = metrics["per_category_iou"]
    Overall_Acc=metrics["overall_accuracy"]
    

    return mean_iou, mean_accuracy, per_category_iou,Overall_Acc