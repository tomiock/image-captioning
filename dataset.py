import csv
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import tokenizers
import torch
from PIL import Image
from pycocotools.coco import COCO
from tokenizers import Encoding, Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import (
    NFD,
    Lowercase,
    StripAccents,
)
from tokenizers.normalizers import (
    Sequence as NormalizerSequence,
)
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VisionDataset


def default_loader(path: str) -> Any:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class COCO_Dataset(VisionDataset):
    def __init__(
        self,
        root,
        annotations_file,
        tokenizer = Optional[Tokenizer],
        max_caption_len = 40,
        transform = None,
    ):
        self.transform = transform
        self.img_folder = root
        self.coco = COCO(annotations_file)
        self.max_caption_len = max_caption_len

        self.ids = list(self.coco.anns.keys())

        # we need the captions to train the tokenizer
        if not tokenizer:
            self.captions = [data['caption'] for data in list(self.coco.anns.values())]

            self.UNK_TOKEN = '<unk>'
            self.PAD_TOKEN = '<pad>'
            self.START_TOKEN = '<start>'
            self.END_TOKEN = '<end>'

            special_tokens = [self.UNK_TOKEN ,self.PAD_TOKEN ,self.START_TOKEN ,self.END_TOKEN]

            self._tokenizer = Tokenizer(WordPiece(unk_token=self.UNK_TOKEN))
            self._tokenizer.normalizer = NormalizerSequence([NFD(), Lowercase(), StripAccents(),])
            self._tokenizer.pre_tokenizer = Whitespace()

            _tokenizer_trainer = WordPieceTrainer(vocab_size=10000, min_frequency=10, special_tokens=special_tokens)
            self._tokenizer.train_from_iterator(self.captions, trainer=_tokenizer_trainer)

            self.pad_token_id = self._tokenizer.token_to_id(self.PAD_TOKEN)
            self._tokenizer.enable_padding(
                length=self.max_caption_len,
                direction='right',
                pad_id=self.pad_token_id,
                pad_token=self.PAD_TOKEN,
            )

            del self.captions # will access them directly later
        else:
            self._tokenizer = tokenizer

        
    @property
    def tokenizer(self) -> tokenizers.Tokenizer:
        return self._tokenizer

    def __getitem__(self, index):
        ann_id = self.ids[index]
        caption = self.coco.anns[ann_id]["caption"]
        img_id = self.coco.anns[ann_id]["image_id"]
        path = self.coco.loadImgs(img_id)[0]["file_name"]

        # Convert image to tensor and pre-process using transform
        image = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # we need to add the special tokens
        caption = self.tokenizer.encode("<start>" + caption + "<end>")

        # the pad token will be added to the right of the end token

        return image, caption

    def __len__(self):
        return len(self.ids)


class Flickr_Dataset(VisionDataset):
    """
    `Flickr8k Entities <http://hockenmaier.cs.illinois.edu/8k-pictures.html>`_ Dataset.
    (Modified to support Kaggle's CSV annotation format)

    Args:
        root (str or ``pathlib.Path``): Root directory where images are located (e.g., 'data/Images/').
        ann_file (string): Path to annotation file (e.g., 'data/captions.txt').
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target (list of captions) and transforms it.
        loader (callable, optional): A function to load an image given its path.
    """

    def __init__(
        self,
        root: str | Path,
        ann_file: str,
        tokenizer: Optional[Tokenizer] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.ann_file = os.path.expanduser(ann_file)
        self.loader = loader

        special_tokens = ["<pad>", "<start>", "<end>", "<unk>"]
        tokenizer = Tokenizer(WordPiece(unk_token="<unk>"))
        tokenizer.normalizer = NormalizerSequence(
            [
                NFD(),
                Lowercase(),
                StripAccents(),
            ]
        )
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordPieceTrainer(
            vocab_size=5000, min_frequency=10, special_tokens=special_tokens
        )

        self.annotations: dict[str, list[str]] = {}
        temp_annotations: dict[str, list[str]] = {}

        with open(self.ann_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)

            _ = next(reader) # take out the header

            for row in reader:
                if len(row) == 2:
                    img_filename, caption = row
                    img_path = os.path.join(str(self.root), img_filename)

                    if img_path not in temp_annotations:
                        temp_annotations[img_path] = []
                    temp_annotations[img_path].append(caption.strip())
                else:
                    pass

        self.annotations = temp_annotations
        self.image_paths = list(sorted(self.annotations.keys()))

        self.all_annontations = [x for xs in self.annotations.values() for x in xs]

    @property
    def list_captions(self):
        return self.all_annontations

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        img_id = self.image_paths[index]

        # Images
        img = self.loader(img_id)
        if self.transform is not None:
            img = self.transform(img)

        # Captions
        target = self.annotations[img_id]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.image_paths)


class EncodeCaptionsTransform:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, captions_text_list: list[str]) -> list[Encoding]:
        encoded_captions: list[Encoding] = []
        for caption_text in captions_text_list:
            preprocessed_text = get_preprocessed_caption(caption_text)
            encoded_captions.append(self.tokenizer.encode(preprocessed_text))
        return encoded_captions


def captioning_collate_fn(batch):
    images = []
    selected_captions = []

    for item in batch:
        image_tensor = item[0]
        list_of_all_captions_for_image = item[1]

        images.append(image_tensor)
        caption = random.choice(list_of_all_captions_for_image)
        caption = torch.tensor(caption.ids)
        selected_captions.append(caption)

    images_batch = torch.stack(images, 0)
    captions_batch = torch.stack(selected_captions, 0)

    return images_batch, captions_batch


def get_preprocessed_caption(caption):
    caption = re.sub(r"\s+", " ", caption)
    caption = caption.strip()
    caption = "<start> " + caption + " <end>"
    return caption


def caption_transforms(
    captions: list[str],
):
    return list(map(get_preprocessed_caption, captions))


def denormalize_image(img: torch.Tensor) -> np.ndarray:
    denormalized_image = img.permute(1, 2, 0).numpy()
    mean = torch.tensor([0.485, 0.456, 0.406]).numpy()
    std = torch.tensor([0.229, 0.224, 0.225]).numpy()
    denormalized_image = std * denormalized_image + mean
    denormalized_image = denormalized_image.clip(0, 1)
    return denormalized_image


if __name__ == "__main__":
    # testing the coco

    image_transforms = transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = COCO_Dataset(
        '/home/datasets/COCO_Dataset/annotations/captions_val2014.json',
        '/home/datasets/COCO_Dataset/val2014/',
        #transform=image_transforms,
    )

    tokenizer = COCO_Dataset.tokenizer

    img, caption = dataset[8123]
    plt.imshow(img)
    plt.axis('off')
    plt.title(caption.tokens)
    plt.show()

    exit()

    # transforms that are needed to run the model by inception_v3

    preliminary_dataset = Flickr_Dataset(
        root="data/Images/",
        ann_file="data/captions.txt",
    )

    all_image_ids_sorted = (
        preliminary_dataset.image_paths
    )  # These are sorted unique image paths
    num_total_images = len(all_image_ids_sorted)

    # Create a list of indices [0, 1, ..., num_total_images-1]
    master_indices = list(range(num_total_images))

    random.seed(42)  # For reproducible splits
    random.shuffle(master_indices)

    # Define split ratios
    train_ratio = 0.7
    val_ratio = 0.15
    # test_ratio is implicitly 1.0 - train_ratio - val_ratio

    train_cutoff = int(train_ratio * num_total_images)
    val_cutoff = int((train_ratio + val_ratio) * num_total_images)

    train_indices = master_indices[:train_cutoff]
    val_indices = master_indices[train_cutoff:val_cutoff]
    test_indices = master_indices[val_cutoff:]

    print(f"Total images: {num_total_images}")
    print(f"Training images: {len(train_indices)}")
    print(f"Validation images: {len(val_indices)}")
    print(f"Test images: {len(test_indices)}")

    train_captions_tokenizer = []
    for idx in train_indices:
        img_path = all_image_ids_sorted[idx]
        train_captions_tokenizer.extend(preliminary_dataset.annotations[img_path])

    special_tokens = ["<pad>", "<start>", "<end>", "<unk>"]
    tokenizer = Tokenizer(WordPiece(unk_token="<unk>"))
    tokenizer.normalizer = NormalizerSequence(
        [
            NFD(),
            Lowercase(),
            StripAccents(),
        ]
    )
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(
        vocab_size=5000, min_frequency=10, special_tokens=special_tokens
    )

    tokenizer.train_from_iterator(train_captions_tokenizer, trainer=trainer)

    pad_token = "<pad>"
    pad_token_id = tokenizer.token_to_id(pad_token)
    tokenizer.enable_padding(
        length=50,
        direction="right",
        pad_id=pad_token_id,
        pad_token=pad_token,
    )

    # our own transform for the captions that uses the tokenizer
    caption_encoder_transform = EncodeCaptionsTransform(tokenizer)

    dataset = Flickr_Dataset(
        root="data/Images/",
        ann_file="data/captions.txt",
        transform=image_transforms,
        target_transform=caption_encoder_transform,
    )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    batch_size = 1
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=0,
        collate_fn=captioning_collate_fn,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=0,
        collate_fn=captioning_collate_fn,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle test data
        num_workers=0,
        collate_fn=captioning_collate_fn,
    )

    for image, caption in train_dataloader:
        denormalized_image = denormalize_image(image.squeeze(0))

        caption = caption.squeeze(0)

        plt.imshow(denormalized_image)
        plt.title(tokenizer.decode(list(caption)))
        plt.axis("off")
        plt.show()
