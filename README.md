### Dataset Download
1. Create directories for the dataset
```
mkdir data/annotations/
mkdir data/images/
```

2. Download the zip files
```
wget ...
wget ...
```

3. Unzip
```
unzip data/Flickr8k_Dataset.zip -d data/images/
unzip data/Flickr8k_text.zip -d data/annotations/
```

4. Delete the "macosx" directories
```
rm -r data/annotations/__MACOSX
rm -r data/images/__MACOSX
```

5. Rename and organize for easier use
```
mv data/images/Flickr8k_Dataset/* data/images
```
> Now the images are located on `data/images/`

