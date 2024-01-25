# How to use

To use this program, you can execute the following command in your terminal:

```bash
python main.py --segformer --batch 16 --classes_to_ignore 0 1 --train
```

This command launches the SegFormer model in training mode with a batch size of 16. It also ignores classes 0 (no_data) and 1 (clouds). 

**!! Don't forget to change the path of your files in the main.py !!**

### Available Arguments:

The program supports various command-line arguments that can be specified using the `arg_parser.py` file. Here are the available arguments:

- `--test`: Start testing mode.
- `--train`: Start training mode.
- `--augmentation`: Enable training with augmentation.
- `--segformer`: Use SegFormer for semantic segmentation.
- `--mit_b3`: Use SegFormer Mit-B3.
- `--mit_b5`: Use SegFormer Mit-B5.
- `--unet`: Use U-Net.
- `--classes_to_ignore` or `-classes_ign`: List of numbers representing classes to ignore (default is [0, 1]).
- `--batch_size` or `-batch`: Set the batch size (default is 16).
- `--save_model` or `-save`: Save checkpoints for the current model.
- `--num_channels` or `-chan`: Set the number of spectral bands (default is 4).
- `--deeplab`: Use Deeplab model.
- `--weighted`: Apply weighted loss.

Here is another exemple :

```bash
python main.py --segformer --batch 16 --classes_to_ignore 0 1 --test
```

*Note that for the test, you may have to change path in `models.py` for the Segformer or `main.py` for the U-Net.*

Make sure to **check the source code** for more details on each argument and its usage.

If you need further assistance or have any questions, don't hesitate to reach out to the project's contributors.

