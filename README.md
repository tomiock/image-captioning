### Dataset Download
1. Create directories for the dataset
```
mkdir data/annotations/
mkdir data/images/
```

2. Download the zip files
```
curl -L -o data/dataset.zip https://www.kaggle.com/api/v1/datasets/download/adityajn105/flickr8k
```

```
mv data/Images/Images/* data/Images/
```

3. Unzip
```
unzip data/dataset.zip
```

4. Remove the zip file
``
rm data/dataset.zip
```


> Now the images are located on `data/Images/`
> With the captions files located on `data/captions.txt`

