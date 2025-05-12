import argparse

import wandb
import torch
import matplotlib.pyplot as plt
from tokenizers import Tokenizer

from dataset import default_loader
from model import CNN_Encoder, RNN_Decoder, get_encoder

parser = argparse.ArgumentParser()
parser.add_argument("image")
args = parser.parse_args()

tokenizer = Tokenizer.from_file('models/tokenizer.json')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image = default_loader(args.image)

inception_v3 = get_encoder()
inception_v3.to(device)

embedding_dim = 512
hidden_dim = 512
vocab_size = tokenizer.get_vocab_size()

encoder = CNN_Encoder(in_dim=2048, embedding_dim=embedding_dim)
decoder = RNN_Decoder(embedding_dim, hidden_dim, vocab_size=vocab_size)

# --- download the artifact from W&B
api = wandb.Api()

encoder.load_state_dict(torch.load('models/encoder.pth', map_location=device))
decoder.load_state_dict(torch.load('models/decoder.pth', map_location=device))
