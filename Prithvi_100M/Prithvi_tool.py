import torch
import yaml

from .Prithvi import MAEEncoder
from pathlib import Path
import functools
import json

with (Path(__file__).parent.parent / 'setting.json').open() as f:
    config = json.load(f)


def load_encoder(frame: int = 1):
    ckpt = config["base checkpoint"]
    yaml_file = Path(__file__).parent / "Prithvi_100M_config.yaml"

    # Load the model configuration from YAML
    with open(yaml_file, "r") as f:
        params = yaml.safe_load(f)

    # Model parameters
    model_params = params["model_args"]
    img_size = model_params["img_size"]
    depth = model_params["depth"]
    patch_size = model_params["patch_size"]
    embed_dim = model_params["embed_dim"]
    num_heads = model_params["num_heads"]
    tubelet_size = model_params["tubelet_size"]

    # Initialize the encoder from the MaskedAutoencoderViT architecture
    encoder = MAEEncoder(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=frame,  # 'frame' can be passed as a function argument
        tubelet_size=tubelet_size,
        in_chans=6,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        norm_layer=functools.partial(torch.nn.LayerNorm, eps=1e-6)
    )

    # Load model weights
    state_dict = torch.load(ckpt)

    # Adjust state_dict to fit the encoder (skip decoder and position embedding for the decoder)
    encoder_state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if 'decoder' not in k}
    del encoder_state_dict["pos_embed"]  # Removing the fixed positional embedding

    # Load the state dict into the encoder, ignoring missing or extra keys
    encoder.load_state_dict(encoder_state_dict, strict=False)

    print(f"Loaded encoder weights from {ckpt}")
    return encoder

