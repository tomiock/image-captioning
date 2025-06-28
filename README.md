# Image Captioning using PyTorch

Some examples to see the model's performance:
![image](https://github.com/user-attachments/assets/71e16e60-2a49-469b-8b98-e7a3430625bc)


### Inference
Provide a directory containing images with the `--root` argument to `inference.py`:
```bash
python3 inference.py --root test_images/
```
The script should cycle between the images displaying -`matplotlib`- them with their corresponding caption.

### Training
There are two scripts for training, on two different datasets.

Due to the model relying on its own embedding layer for the text representation, the `flickr30k` dataset is insufficient for training the model.

Using the MS COCO dataset, the model convergences rapidly showing good results. Trained for 10 epochs during 3h on a 12GB A40.

```bash
python3 train_coco.py
```
Use `wandb disable` beforehand to disable the logging.

IMPORTANT: The dataset root directoy should be changed from the script.
