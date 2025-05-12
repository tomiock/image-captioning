### Dataset Download
1. Create directories for the dataset
```
mkdir data/images/
```

2. Download the zip files
```
curl -L -o data/dataset.zip https://www.kaggle.com/api/v1/datasets/download/adityajn105/flickr8k
```

Extract the zip:
```
unzip data/dataset.zip
```

Move the images folder
```
mv Images/ data/
```

```
mv captions.txt data/
```

3. Create the test dataset [DO ONCE]
**Skip if already have `test_captions.txt` in the `data` folder.
```
python3 data/create_test_captions.py
```

4. Using the COMMITTED test captions file, we recreate the test dataset

Update the captions:
```
python3 data/update_captions.py
```
Update the images:
```
python3 data/move_test_images.py
```

5. Remove the zip file
```
rm data/dataset.zip
```

> Now the images are located on `data/Images/`
> With the captions files located on `data/captions.txt`
