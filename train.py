import os
import random

import sys
import numpy as np

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
from model import CNN_Encoder, RNN_Decoder

np.set_printoptions(threshold=sys.maxsize)

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
    train_ratio = 0.8

    train_cutoff = int(train_ratio * num_total_images)

    train_indices = master_indices[:train_cutoff]
    val_indices = master_indices[train_cutoff:]

    print(f"Total images: {num_total_images}")
    print(f"Training images: {len(train_indices)}")
    print(f"Validation images: {len(val_indices)}")

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

    # train and save the tokenizer
    tokenizer.train_from_iterator(train_captions_tokenizer, trainer=trainer)
    tokenizer.save("models/tokenizer.json")

    pad_token = "<pad>"
    pad_token_id = tokenizer.token_to_id(pad_token)
    tokenizer.enable_padding(
        length=200,
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

    run = wandb.init(
        entity="uni-DL-2025",
        project="image-captioning",
        config={
            "learning_rate": 0.0001,
            "epochs": 10,
            "batch_size": 256,
            "embedding_dim": 512,
            "hidden_dim": 512,
            "vocab_size": tokenizer.get_vocab_size(),
            "scheduler": "CosineAnnealingLR",
        },
    )

    batch_size = wandb.config.batch_size
    n_workers = 10
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        collate_fn=captioning_collate_fn,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=n_workers,
        collate_fn=captioning_collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # --- MODEL Declarations ---

    # we loaded it with default pretrained weights
    wandb.define_metric("epoch")
    wandb.define_metric("train/epoch_loss", step_metric="epoch")
    wandb.define_metric("val/epoch_loss", step_metric="epoch")
    wandb.define_metric("lr", step_metric="epoch")
    wandb.define_metric("train_batch_step")
    wandb.define_metric("train/batch_loss", step_metric="train_batch_step")

    vocab_size: int = wandb.config.vocab_size

    embedding_dim = wandb.config.embedding_dim
    hidden_dim = wandb.config.hidden_dim

    encoder = CNN_Encoder(embed_size=embedding_dim)
    decoder = RNN_Decoder(embedding_dim, hidden_dim, vocab_size=vocab_size)

    use_latest_model = False
    if use_latest_model:
        en_artifact = run.use_artifact("encoder:latest")
        de_artifact = run.use_artifact("decoder:latest")

        en_dir = en_artifact.download()
        de_dir = de_artifact.download()

        en_path = os.path.join(en_dir, "encoder.pth")
        de_path = os.path.join(de_dir, "decoder.pth")

        encoder.load_state_dict(torch.load(en_path))
        decoder.load_state_dict(torch.load(de_path))

        tok_artifact = run.use_artifact("tokenizer:latest")
        tok_dir = tok_artifact.download()
        toke_path = os.path.join(tok_dir, "tokenizer.json")

        tokenizer = Tokenizer.from_file(toke_path)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # we only want to train the encoder and decoder at the moment
    total_params = list(encoder.parameters()) + list(decoder.parameters())

    # --- LOSS and Optimizer ---
    learning_rate = wandb.config.learning_rate
    criterion = torch.nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(total_params, lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=wandb.config.epochs, eta_min=1e-6
    )

    # --- TRAINING ---
    num_epochs = wandb.config.epochs

    train_batch_step = 0
    best_val_loss = np.inf

    wandb.watch(encoder, criterion, log="all", log_freq=100, log_graph=True)
    wandb.watch(decoder, criterion, log="all", log_freq=100, log_graph=True)

    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        total_train_loss = 0.0

        train_pbar = tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [TRAIN]"
        )
        for images, captions in tqdm(train_dataloader):
            captions = captions.to(device)
            images = images.to(device)

            optimizer.zero_grad()

            images_encoded = encoder(images)

            outputs = decoder(images_encoded, captions)

            outputs = outputs[:, :-1, :].reshape(-1, vocab_size)
            captions = captions[:, 1:].reshape(-1)
            
            loss = criterion(outputs, captions)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_batch_step += 1
            wandb.log(
                {
                    "train/batch_loss": loss.item(),
                    "train_batch_step": train_batch_step,
                }
            )
            wandb.log({"train/batch_loss": loss.item()})
            train_pbar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_dataloader)
        wandb.log({"train/loss": avg_train_loss, "epoch": epoch + 1})

        encoder.eval()
        decoder.eval()
        total_val_loss = 0.0

        val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [VAL]")
        with torch.no_grad():
            for images_val, captions_val in val_pbar:
                captions_val = captions_val.to(device)
                images_val = images_val.to(device)

                images_features_val = encoder(images_val)

                pred = decoder(images_features_val, captions_val)

                val_loss = criterion(pred.view(-1, vocab_size), captions_val.view(-1))

                total_val_loss += val_loss.item()
                val_pbar.set_postfix(loss=val_loss.item())

        avg_val_loss = total_val_loss / len(val_dataloader)
        wandb.log({"val/epoch_loss": avg_val_loss, "epoch": epoch + 1})

        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log({"lr": current_lr, "epoch": epoch + 1})

        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(decoder.state_dict(), "models/decoder.pth")
            torch.save(encoder.state_dict(), "models/encoder.pth")

        print(
            f"Epoch {epoch + 1}/{num_epochs} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}"
        )

    wandb.log({"best_val_loss": best_val_loss})

    tokenizer.save("models/tokenizer.json")

    artifact_decoder = wandb.Artifact("decoder", type="model")
    artifact_decoder.add_file("models/decoder.pth")
    run.log_artifact(artifact_decoder)

    artifact_encoder = wandb.Artifact("encoder", type="model")
    artifact_encoder.add_file("models/encoder.pth")
    run.log_artifact(artifact_encoder)

    artifact_tokenizer = wandb.Artifact("tokenizer", type="model")
    artifact_tokenizer.add_file("models/tokenizer.json")
    run.log_artifact(artifact_tokenizer)

    wandb.finish()
