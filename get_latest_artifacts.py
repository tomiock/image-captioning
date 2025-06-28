import wandb

api = wandb.Api()

encoder_artifact = api.artifact('uni-DL-2025/image-captioning/encoder_large:latest')
decoder_artifact = api.artifact('uni-DL-2025/image-captioning/decoder_large:latest')
tokenizer_artifact = api.artifact('uni-DL-2025/image-captioning/tokenizer_large:latest')


encoder_artifact.download(root='models/')
decoder_artifact.download(root='models/')
tokenizer_artifact.download(root='models/')
