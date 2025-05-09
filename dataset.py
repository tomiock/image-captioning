import csv
import os
import random
import re
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from tokenizers import Tokenizer, Encoding
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


class Flickr8kDataset(VisionDataset):
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
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.ann_file = os.path.expanduser(ann_file)
        self.loader = loader

        self.annotations: dict[str, list[str]] = {}
        temp_annotations: dict[str, list[str]] = {}
        with open(self.ann_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)

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
        self.ids = list(sorted(self.annotations.keys()))

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
        img_id = self.ids[index]

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
        return len(self.ids)


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
    # transforms that are needed to run the model by inception_v3
    image_transforms = transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    preliminary_dataset = Flickr8kDataset(
        root="data/Images/",
        ann_file="data/captions.txt",
    )

    all_image_ids_sorted = (
        preliminary_dataset.ids
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

    dataset = Flickr8kDataset(
        root="data/Images/",
        ann_file="data/captions.txt",
        transform=image_transforms,
        target_transform=caption_encoder_transform,
    )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    batch_size = 12
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

    print(f"Tokenizer Vocabulary Size: {tokenizer.get_vocab_size()}")

    print("\n--- Example from Dataset (dataset[0]) ---")
    image_example, list_of_encodings_example = train_dataset[0]
    print(f"Image tensor shape: {image_example.shape}")
    print(f"Number of captions for this image: {len(list_of_encodings_example)}")
    if list_of_encodings_example:
        example_encoding = list_of_encodings_example[0]
        print(f"Example caption tokens: {example_encoding.tokens}")
        print(f"Example caption IDs (padded): {example_encoding.ids}")

    print("\n--- Example from DataLoader (batch) ---")
    try:
        denormalized_image = denormalize_image(image_example)

        plt.imshow(denormalized_image)
        if list_of_encodings_example:
            plt.title(
                tokenizer.decode(
                    list_of_encodings_example[0].ids, skip_special_tokens=True
                )
            )
        plt.axis("off")
        plt.show()

    except StopIteration:
        print(
            "DataLoader is empty. This might happen if the dataset is too small for the batch size."
        )
    except IndexError:
        print(
            "IndexError during data loading or processing. Dataset might be empty or an image is missing."
        )
