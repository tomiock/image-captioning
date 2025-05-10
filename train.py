import random

import wandb
import torch
from tokenizers import Tokenizer
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
from tqdm import tqdm

from dataset import Flickr8kDataset, captioning_collate_fn, EncodeCaptionsTransform
from model import get_encoder, CNN_Encoder, RNN_Decoder

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

    batch_size = 128
    n_workers = 10
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        collate_fn=captioning_collate_fn,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        collate_fn=captioning_collate_fn,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        collate_fn=captioning_collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- MODEL Declarations ---

    # we loaded it with default pretrained weights
    inception_v3 = get_encoder()
    inception_v3.to(device)

    wandb.init(
        entity='uni-DL-2025',
        project='image-captioning',
    )

    vocab_size: int = tokenizer.get_vocab_size()

    embedding_dim = hidden_dim = 512
    encoder = CNN_Encoder(in_dim=2048, embedding_dim=embedding_dim)
    decoder = RNN_Decoder(embedding_dim, hidden_dim, vocab_size=vocab_size)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # we only want to train the encoder and decoder at the moment
    total_params = list(encoder.parameters()) + list(decoder.parameters())

    # --- LOSS and Optimizer ---
    learning_rate = .0001

    criterion = torch.nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(total_params, lr=learning_rate)


    # --- TRAINING ---
    num_epochs = 10

    for epoch in range(num_epochs):
        for images, captions in tqdm(train_dataloader):
            captions = captions.to(device)
            images = images.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                images_features, _ = inception_v3(images)
            
            images_encoded = encoder(images_features)

            outputs = decoder(images_encoded, captions)

            # needs to be a mean
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
            loss.backward()
            optimizer.step()

            wandb.log({"train/batch_loss": loss.item()})

