# How to download the models from W&B

The idea is that the models are stored in the weights and biases server. That way we can train on the VM and then perform inference on our local machines easily.

### Commands to use
Download the encoder:
```
wandb artifact get uni-DL-2025/image-captioning/encoder:v1 --root models/
```

Download the decoder:
```
wandb artifact get uni-DL-2025/image-captioning/decoder:v1 --root models/
```

