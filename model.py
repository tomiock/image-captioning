import torch
import torch.nn as nn

from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from torchinfo import summary

from torchvision.models.vision_transformer import ViT_B_16_Weights


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
            embedding_dim,
            hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
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


def get_target_mask(size: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(size, size) * float("-inf"))
    return mask.float()


class TransformerDecoder(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, vocab_size: int, num_layers: int
    ):
        super(TransformerDecoder, self).__init__()

        MAX_LENGHT = 512
        self.pos_encoding = nn.Parameter(torch.randn(1, MAX_LENGHT, hidden_size))

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            batch_first=True,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=num_layers,
        )

        self.linear_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, image_tokens, text_sequence) -> torch.Tensor:
        # text_sequence: (B, L)
        # image_tokens: (B, N, D)

        text_embeddings = self.embedding(text_sequence)  # (B, L, D)

        text_embeddings = ( # we need to set the PE with the text shape
            text_embeddings + self.pos_encoding[:, : text_embeddings.shape[1], :]
        )  # (B, L, D)

        target_mask = get_target_mask(text_sequence.shape[1]).to(text_sequence.device)

        output = self.decoder(
            tgt=text_embeddings, memory=image_tokens, tgt_mask=target_mask
        )  # (L, B, D)

        output = self.linear_head(output)  # (B, L, vocab_size)

        return output

    def sample(self, image_tokens, start_token_id, end_token_id, max_length=50):
        # image_tokens: (B, N, D)

        device = image_tokens.device
        batch_size = image_tokens.shape[0]

        generated_sequence = torch.full(
            (batch_size, 1), start_token_id, dtype=torch.long, device=device
        )

        for _ in range(max_length):
            # Embed the current generated sequence
            tgt_embeddings = self.embedding(generated_sequence)  # (B, L_gen, D)

            tgt_embeddings = (
                tgt_embeddings + self.pos_encoding[:, : tgt_embeddings.shape[1], :]
            )  # (B, L_gen, D)

            tgt_embeddings = tgt_embeddings.permute(1, 0, 2)  # (L_gen, B, D)

            target_mask = get_target_mask(generated_sequence.shape[1]).to(device)

            output = self.decoder(
                tgt=tgt_embeddings, memory=image_tokens, tgt_mask=target_mask
            )  # (L_gen, B, D)

            last_token_output = output[-1, :, :]  # (B, D)
            logits = self.linear_head(last_token_output)  # (B, vocab_size)

            next_token = torch.argmax(logits, dim=-1)  # (B)

            generated_sequence = torch.cat(
                (generated_sequence, next_token.unsqueeze(1)), dim=1
            )  # (B, L_gen + 1)

            if torch.all(next_token == end_token_id):
                break

        return generated_sequence


class ImageCaptioningModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        frozen_vit: bool = True,
    ):
        super(ImageCaptioningModel, self).__init__()
        self.encoder, _ = get_vit_encoder(frozen=frozen_vit)
        self.decoder = TransformerDecoder(
            hidden_size, num_heads, vocab_size, num_layers
        )

        self.image_token_proj = nn.Linear(768, hidden_size, bias=False)

    def forward(self, images, text_sequence) -> torch.Tensor:
        # images: (B, C, H, W)
        # text_sequence: (B, L)

        encoder_output = self.encoder(images)
        image_tokens = encoder_output["visual_tokens"]  # (B, N+1, D)
        image_tokens = self.image_token_proj(image_tokens)

        output = self.decoder(image_tokens, text_sequence)  # (B, L, vocab_size)
        return output

    def sample(self, images, start_token_id, end_token_id, max_length=50):
        # images: (B, C, H, W)

        encoder_output = self.encoder(images)
        image_tokens = encoder_output["visual_tokens"]  # (B, N+1, D)

        generated_sequence = self.decoder.sample(
            image_tokens, start_token_id, end_token_id, max_length
        )
        return generated_sequence


def get_vit_encoder(frozen=False):
    """
    Input image size: (384, 384)

    Output size: (B, N+1, 768)
    """
    weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    vit_model = models.vit_b_16(weights)

    vit_model = create_feature_extractor(
        vit_model, return_nodes={"encoder.ln": "visual_tokens"}
    )

    # by default we do NOT train the model
    for param in vit_model.parameters():
        param.requires_grad = frozen

    return vit_model, weights


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
