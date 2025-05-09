import torch
import torch.nn as nn

from torchvision.models import Inception_V3_Weights
from torchinfo import summary


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class CNN_Encoder(nn.Module):
    def __init__(self, in_dim, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.flat = nn.Flatten()
        self.fc = nn.Linear(in_dim, embedding_dim)

    def forward(self, x):
        x = self.flat(x)
        x = self.fc(x)
        x = nn.functional.relu(x)
        return x


class RNN_Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.hidden_dim = hidden_dim

        # on the original paper units=embed_dim=512

        self.embedding = nn.Embedding(vocab_size, embedding_dim, 0)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.relu = nn.ReLU()

    def forward(self, features, captions):
        embed = self.embedding(captions[:, :-1])  # (B, S, embed_dim)

        features = features.unsqueeze(1)

        lstm_input = torch.cat((features, embed), dim=1)

        output, _ = self.lstm(lstm_input)

        # (B, S, vocab_size)
        output = self.fc(output)

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

    model.dropout = Identity()
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
