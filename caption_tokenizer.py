from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import (
    NFD,
    Lowercase,
    StripAccents,
    Sequence as NormalizerSequence,
)

sample_captions = [
    "A cat sits on the mat.",
    "The quick brown fox jumps over the lazy dog.",
    "Another example sentence for our tokenizer.",
    "Tokenization is fun and useful!",
    "A cat can also jump, not just sit.",
]

special_tokens = ["<pad>", "<start>", "<end>"]

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
    vocab_size=100, min_frequency=1, special_tokens=special_tokens
)

tokenizer.train_from_iterator(sample_captions, trainer=trainer)

pad_token = "<pad>"
pad_token_id = tokenizer.token_to_id(pad_token)
tokenizer.enable_padding(
    length=200,
    direction="right",
    pad_id=pad_token_id,
    pad_token=pad_token,
)

test_sentence = "<start> A quick cat jumps. <end>"
print(f"\nOriginal sentence: '{test_sentence}'")

encoded_output = tokenizer.encode(test_sentence)
print(encoded_output.tokens)
