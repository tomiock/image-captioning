import os
import argparse
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



embedding_dim = 1028
hidden_dim = 1028
vocab_size = tokenizer.get_vocab_size()

END_TOKEN = '<end>'
end_token_id = tokenizer.token_to_id(END_TOKEN)

encoder = CNN_Encoder(embed_size=embedding_dim)
decoder = RNN_Decoder(embedding_dim, hidden_dim, vocab_size=vocab_size, end_token_idx=end_token_id)

encoder.eval()
decoder.eval()

# --- download the artifact from W&B
api = wandb.Api()

encoder.load_state_dict(torch.load("models/encoder.pth", map_location=device))
decoder.load_state_dict(torch.load("models/decoder.pth", map_location=device))

parser = argparse.ArgumentParser()
parser.add_argument("--root", default="test")
args = parser.parse_args()

if args.root == 'all':
    ROOT_PATH = "data/coco/test2014"
else:
    ROOT_PATH = "data/coco/test_images"

for path in os.listdir(ROOT_PATH):
    image = os.path.join(ROOT_PATH, path)
    image = default_loader(image)

    tensor_image = image_transforms(image)
    encoded_features = encoder(tensor_image.unsqueeze(0).to(device))
    output = decoder.sample(encoded_features)
    caption = tokenizer.decode(output)

    plt.imshow(image)
    plt.title(caption)
    plt.show()
