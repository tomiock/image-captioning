import os
import random

import wandb
import torch
import matplotlib.pyplot as plt
from tokenizers import Tokenizer

from dataset import default_loader
from model import CNN_Encoder, RNN_Decoder
import torchvision.transforms as transforms

tokenizer = Tokenizer.from_file("models/tokenizer.json")

image_transforms = transforms.Compose(
    [
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random_image = random.choice(os.listdir("data/test_images/"))
random_image = os.path.join("data/test_images", random_image)

image = default_loader(random_image)

embedding_dim = 512
hidden_dim = 512
vocab_size = tokenizer.get_vocab_size()

encoder = CNN_Encoder(embed_size=embedding_dim)
decoder = RNN_Decoder(embedding_dim, hidden_dim, vocab_size=vocab_size)

# --- download the artifact from W&B
api = wandb.Api()

encoder.load_state_dict(torch.load("models/encoder.pth", map_location=device))
decoder.load_state_dict(torch.load("models/decoder.pth", map_location=device))

tensor_image = image_transforms(image)

encoded_features = encoder(tensor_image.unsqueeze(0).to(device))

output = decoder.sample(encoded_features)
caption = tokenizer.decode(output)

plt.imshow(image)
plt.title(caption)
plt.show()
