import re
import random
import torch
import nltk
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.models import Inception_V3_Weights
from torchvision import transforms
from torchtext.data import get_tokenizer

import os
import csv
from pathlib import Path
from typing import Any, Optional, Callable
from collections import Counter

from PIL import Image


def default_loader(path: str) -> Any:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class Flickr8k(VisionDataset):
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


def captioning_collate_fn(batch):
    images = []
    selected_captions = []

    for item in batch:
        image_tensor = item[0]
        list_of_all_captions_for_image = item[1]

        images.append(image_tensor)
        selected_captions.append(random.choice(list_of_all_captions_for_image))

    images_batch = torch.stack(images, 0)

    return images_batch, selected_captions


def get_preprocessed_caption(caption):
    caption = re.sub(r"\s+", " ", caption)
    caption = caption.strip()
    caption = "<start> " + caption + " <end>"
    return caption


def caption_transforms(
    captions: list[str],
):
    return list(map(get_preprocessed_caption, captions))


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

    dataset = Flickr8k(
        root="data/Images/",
        ann_file="data/captions.txt",
        transform=image_transforms,
    )

    flickr_dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=12,
        collate_fn=captioning_collate_fn,
    )

    # it should get the most up to date weights
    model = torch.hub.load(
        "pytorch/vision:v0.10.0", "inception_v3", weights=Inception_V3_Weights.DEFAULT
    )
    model.eval()

    image, captions = dataset[0]

    for caption in captions:
        print(caption)

    input_batch = image.unsqueeze(0)  # create a mini-batch as expected by the model

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    with open("data/imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())

    plt.imshow(torch.permute(image, (1, 2, 0)))
    plt.show()

