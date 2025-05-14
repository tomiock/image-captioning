import wandb

api = wandb.Api()

encoder_artifact = api.artifact('uni-DL-2025/image-captioning/encoder:latest')
decoder_artifact = api.artifact('uni-DL-2025/image-captioning/decoder:latest')
tokenizer_artifact = api.artifact('uni-DL-2025/image-captioning/tokenizer:latest')


encoder_artifact.download(root='models/')
decoder_artifact.download(root='models/')
tokenizer_artifact.download(root='models/')
