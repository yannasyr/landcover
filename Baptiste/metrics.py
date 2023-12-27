import numpy as np
import torch
import matplotlib as plt
from LandCoverData import LandCoverData



# Estimation du vecteur Y = [no_data, clouds, artificial, cultivated, broadleaf, coniferous, herbaceous, natural, snow, water]
def get_Y(mask2d):
  occurrences = np.bincount(mask2d.flatten(), minlength=10)
  Y = occurrences / np.sum(occurrences)
  return Y

def klmetric(y_t, y_p, eps):
  sum = 0
  for i in range(len(y_t)):
    sum += (y_t[i]+eps)*np.log((y_t[i] + eps) / (y_p[i] + eps))
  return sum

def mesure_on_batch(batch_gt, batch_predi, batch_size=16):
  mean = 0
  for i in range(batch_size):
    Y_pred = get_Y(batch_gt[i].cpu().numpy())
    Y_truth = get_Y(batch_predi[i].cpu().numpy())
    mean += klmetric(Y_truth, Y_pred, 10e-8)
  return mean / batch_size

def mesure_on_dataloader(data_loader,device,model):

  mean = 0
  for i in range(len(data_loader)):
    test_inputs, test_targets = next(iter(data_loader))
    test_pixels_values = test_inputs.to(device)
    model.eval()
    with torch.no_grad():
      test_outputs = model(pixel_values=test_pixels_values)
      test_logits = test_outputs.logits

    _, predicted_labels = torch.max(test_logits, dim=1)

    mean += mesure_on_batch(test_targets, predicted_labels)

  return mean / len(data_loader)



def affichage(model,data_loader,device):
  CLASSES_COLORPALETTE=LandCoverData.CLASSES_COLORPALETTE

  # Assuming val_loader is defined in your code
  val_inputs, val_targets = next(iter(data_loader))

  val_pixel_values = val_inputs.to(device)

  # Set the model to evaluation mode
  model.eval()
  # Perform inference on the validation image
  with torch.no_grad():
      val_outputs = model(pixel_values=val_pixel_values)
      val_logits = val_outputs.logits

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
      return 