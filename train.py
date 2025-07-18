import os
import random
import numpy as np

import wandb
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import COCO_Dataset
from model import ImageCaptioningModel
from torchvision.models import ViT_B_16_Weights

# --- CORRECTED ViT Weights and Transforms Initialization ---
vit_weights = ViT_B_16_Weights.IMAGENET1K_V1
image_transforms = vit_weights.transforms()

train_dataset = COCO_Dataset(
    root="/home/datasets/COCO_Dataset/train2014",
    annotations_file="/home/datasets/COCO_Dataset/annotations/captions_train2014.json",
    max_caption_len=80,
    tokenizer=None,
    transform=image_transforms,
)

tokenizer = train_dataset.tokenizer

val_dataset = COCO_Dataset(
    root="/home/datasets/COCO_Dataset/val2014",
    annotations_file="/home/datasets/COCO_Dataset/annotations/captions_val2014.json",
    max_caption_len=80,
    tokenizer=tokenizer,
    transform=image_transforms,
)


def captioning_collate_fn(batch):
    images = []
    captions = []

    for item in batch:
        image_tensor = item[0]
        caption = item[1]

        images.append(image_tensor)
        caption = torch.tensor(caption.ids)
        captions.append(caption)

    images_batch = torch.stack(images, 0)
    captions_batch = torch.stack(captions, 0)

    return images_batch, captions_batch


# --- Get Special Token IDs ---
start_token_id = tokenizer.token_to_id("<start>")
end_token_id = tokenizer.token_to_id("<end>")
pad_token_id = tokenizer.token_to_id("<pad>")
unk_token_id = tokenizer.token_to_id("<unk>")

run = wandb.init(
    entity="uni-DL-2025",
    project="image-captioning",
    config={
        "learning_rate": 0.0001,
        "epochs": 1,
        "batch_size": 52,
        "embedding_dim": 512,
        "hidden_dim": 512,
        "vocab_size": tokenizer.get_vocab_size(),
        "scheduler": "CosineAnnealingLR",
        "start_token_id": start_token_id,
        "end_token_id": end_token_id,
        "pad_token_id": pad_token_id,
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
wandb.define_metric("epoch")
wandb.define_metric("train/epoch_loss", step_metric="epoch")
wandb.define_metric("val/epoch_loss", step_metric="epoch")
wandb.define_metric("lr", step_metric="epoch")
wandb.define_metric("train_batch_step")
wandb.define_metric("train/batch_loss", step_metric="train_batch_step")
wandb.define_metric("val_batch_step")
wandb.define_metric("val/batch_loss", step_metric="val_batch_step")
wandb.define_metric("sampled_loss", step_metric="train_batch_step") # New metric for sampled images' loss

vocab_size: int = wandb.config.vocab_size

embedding_dim = wandb.config.embedding_dim
hidden_dim = wandb.config.hidden_dim

model = ImageCaptioningModel(
    hidden_size=hidden_dim, vocab_size=vocab_size, num_heads=4, num_layers=4
)

model.to(device)

# --- LOSS and Optimizer ---
learning_rate = wandb.config.learning_rate
criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=wandb.config.epochs, eta_min=1e-6
)

# --- INFERENCE FUNCTION for Logging Samples ---
def log_sampled_captions(
    model: nn.Module,
    tokenizer,
    dataset: COCO_Dataset,
    device: torch.device,
    criterion: torch.nn.Module, # Pass criterion to calculate loss
    num_samples: int = 10,
    max_decode_length: int = 80,
    log_name: str = "sampled_captions",
    current_step: int = 0,
):
    model.eval() # Set model to evaluation mode
    
    # Pick random indices
    indices = random.sample(range(len(dataset)), num_samples)
    
    wandb_images = []
    total_sampled_loss = 0.0 # To accumulate loss for sampled images
    
    with torch.no_grad():
        for idx in indices:
            # Get image and original caption from dataset
            img_tensor, original_tokenized_caption = dataset[idx]
            
            # Reshape image for model input (add batch dimension)
            img_input = img_tensor.unsqueeze(0).to(device) # (1, C, H, W)
            
            # Prepare original caption for teacher-forced loss calculation
            # Unsqueeze to add batch dimension (1, L)
            original_caption_tensor = torch.tensor(original_tokenized_caption.ids).unsqueeze(0).to(device)

            # --- Calculate loss for this sampled image with ground truth ---
            # Input to model for loss is original caption excluding last token
            loss_outputs = model(img_input, original_caption_tensor[:, :-1]) 
            loss_targets = original_caption_tensor[:, 1:]

            # Reshape for CrossEntropyLoss
            loss = criterion(loss_outputs.reshape(-1, vocab_size), loss_targets.reshape(-1))
            total_sampled_loss += loss.item()

            # --- Generate caption (inference) for display ---
            generated_token_ids = model.sample(
                img_input, start_token_id, end_token_id, max_decode_length
            )
            
            # Decode generated caption
            generated_caption_ids = generated_token_ids[0].tolist()
            filtered_gen_ids = [
                token_id for token_id in generated_caption_ids 
                if token_id not in [start_token_id, end_token_id, pad_token_id]
            ]
            generated_text = tokenizer.decode(filtered_gen_ids)

            # Decode original caption for comparison
            filtered_orig_ids = [
                token_id for token_id in original_tokenized_caption.ids
                if token_id not in [start_token_id, end_token_id, pad_token_id]
            ]
            original_text = tokenizer.decode(filtered_orig_ids)

            wandb_images.append(
                wandb.Image(
                    img_tensor, 
                    caption=f"Step {current_step} | Gen: {generated_text}\nOrig: {original_text}\nLoss: {loss.item():.4f}"
                )
            )
    
    avg_sampled_loss = total_sampled_loss / num_samples
    wandb.log({log_name: wandb_images, "sampled_loss": avg_sampled_loss, "train_batch_step": current_step})
    
    model.train() # Set model back to train mode

# --- TRAINING ---
num_epochs = wandb.config.epochs

train_batch_step = 0
val_batch_step = 0 
best_val_loss = np.inf

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0

    train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [TRAIN]")
    for i, (images, captions) in enumerate(train_pbar):
        captions = captions.to(device)
        images = images.to(device)

        optimizer.zero_grad()

        outputs = model(images, captions[:, :-1])
        targets = captions[:, 1:]

        loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        train_batch_step += 1
        
        # Log training batch loss and sampled images/loss every 100 steps
        if (i + 1) % 100 == 0:
            wandb.log(
                {
                    "train/batch_loss": loss.item(),
                    "train_batch_step": train_batch_step,
                }
            )
            print(f"\nLogging sampled captions and their GT loss at training step {train_batch_step}...")
            log_sampled_captions(
                model, 
                tokenizer, 
                val_dataset, # Using val_dataset for sampling to monitor generalization
                device, 
                criterion, # Pass the criterion
                num_samples=10,
                max_decode_length=80,
                log_name="sampled_captions_at_train_step",
                current_step=train_batch_step
            )
            print(f"Sampled captions and loss logged for step {train_batch_step}.")

        train_pbar.set_postfix(loss=loss.item())

    avg_train_loss = total_train_loss / len(train_dataloader)
    wandb.log({"train/epoch_loss": avg_train_loss, "epoch": epoch + 1})

    torch.save(model.state_dict(), "models/model.pth")
    tokenizer.save("models/tokenizer.json")

    model.eval()
    total_val_loss = 0.0

    val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [VAL]")
    with torch.no_grad():
        for i_val, (images_val, captions_val) in enumerate(val_pbar):
            captions_val = captions_val.to(device)
            images_val = images_val.to(device)

            outputs_val = model(images_val, captions_val[:, :-1])
            targets_val = captions_val[:, 1:]

            val_loss = criterion(outputs_val.reshape(-1, vocab_size), targets_val.reshape(-1))

            total_val_loss += val_loss.item()
            val_batch_step += 1
            
            # Log validation batch loss every 100 steps
            if (i_val + 1) % 100 == 0:
                wandb.log(
                    {
                        "val/batch_loss": val_loss.item(),
                        "val_batch_step": val_batch_step,
                    }
                )
            val_pbar.set_postfix(loss=val_loss.item())

    avg_val_loss = total_val_loss / len(val_dataloader)
    wandb.log({"val/epoch_loss": avg_val_loss, "epoch": epoch + 1})

    current_lr = optimizer.param_groups[0]["lr"]
    wandb.log({"lr": current_lr, "epoch": epoch + 1})

    scheduler.step()

    print(
        f"Epoch {epoch + 1}/{num_epochs} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}"
    )

wandb.log({"best_val_loss": best_val_loss})

artifact_model = wandb.Artifact("model", type="model")
artifact_model.add_file("models/model.pth")
run.log_artifact(artifact_model)

artifact_tokenizer = wandb.Artifact("tokenizer", type="model")
artifact_tokenizer.add_file("models/tokenizer.json")
run.log_artifact(artifact_tokenizer)

wandb.finish()
