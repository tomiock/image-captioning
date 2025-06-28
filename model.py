import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torchvision import models
from torchinfo import summary

from torch.nn.functional import softmax


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class CNN_Encoder(nn.Module):
    def __init__(self, embed_size):
        super(CNN_Encoder, self).__init__()
        # pretrained model resnet50
        resnet = models.resnet50(pretrained=True)

        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # replace the classifier with a fully connected embedding layer
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class RNN_Decoder(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        vocab_size,
        end_token_idx=2,
        number_layers=1,
    ):
        super(RNN_Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = number_layers
        self.end_token = end_token_idx

        assert self.num_layers > 0

        # on the original paper units=embed_dim=512

        self.embedding = nn.Embedding(vocab_size, embedding_dim, 0)

        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=self.num_layers, batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, features, captions):
        embed = self.embedding(captions[:, :-1])  # (B, S, embed_dim)

        features = features.unsqueeze(1)

        lstm_input = torch.cat((features, embed), dim=1)

        output, _ = self.lstm(lstm_input)

        # (B, S, vocab_size)
        output = self.fc(output)

        return output

    def sample(self, features, max_len=20):
        features = features.unsqueeze(0)
        output = []
        (h, c) = (
            torch.randn(self.num_layers, 1, self.hidden_dim),
            torch.randn(self.num_layers, 1, self.hidden_dim),
        )

        x, (h, c) = self.lstm(features, (h, c))
        input = torch.tensor([0])
        for _ in range(max_len):
            token_embed = self.embedding(input)
            token_embed = token_embed.unsqueeze(1)
            lstm_out, (h, c) = self.lstm(token_embed, (h, c))

            x = self.fc(lstm_out.squeeze(1))

            _, pred = x.max(dim=1)

            end_token_predicted = pred.item() == self.end_token
            if end_token_predicted:
                break

            output.append(pred.item())
            input = pred

        return output

    def reset_state(self, batch_size):
        return torch.zeros((batch_size, self.hidden_dim))


def get_encoder() -> nn.Module:
    """
    inception_v3 model that gives us the features that are given to the CNN encoder

    Output size: (B, 2048)
    """
    model: nn.Module = torch.hub.load(
        "pytorch/vision:v0.10.0", "inception_v3", weights=Inception_V3_Weights.DEFAULT
    )

    model.fc = Identity()

    return model


if __name__ == "__main__":
    inception_module: nn.Module = get_encoder()

    batch_size = 2
    summary(inception_module, (batch_size, 3, 299, 299))

    embedding_dim = hidden_dim = 512
    encoder = CNN_Encoder(in_dim=2048, embedding_dim=embedding_dim)
    decoder = RNN_Decoder(embedding_dim, hidden_dim, vocab_size=1000)

    summary(encoder, [(batch_size, 2048)])
    summary(decoder, [(batch_size, 512), (batch_size, 200)])
