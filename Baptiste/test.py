import torch 




model=
test_loader=

# Color palette
  CLASSES_COLORPALETTE = {
      0: [0, 0, 0],
      1: [255, 25, 236],
      2: [215, 25, 28],
      3: [211, 154, 92],
      4: [33, 115, 55],
      5: [21, 75, 35],
      6: [118, 209, 93],
      7: [130, 130, 130],
      8: [255, 255, 255],
      9: [43, 61, 255]
  }

  # Assuming val_loader is defined in your code
  val_inputs, val_targets = next(iter(val_loader))

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




  print(mesure_on_dataloader(val_loader))