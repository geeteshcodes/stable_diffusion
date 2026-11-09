"""
Inference & Image Generation Script for Text-to-Image Diffusion U-Net (v2).
Loads trained weights (locally or automatically from Hugging Face Hub: geeteshcodes/t2i-coco-128-clipL)
and samples synthetic images using Classifier-Free Guidance (CFG) with DDIM sampling.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import tensorflow as tf

from architecture import TextToImageUNet
from diffusion import GaussianDiffusion
from text_encoder import TextEncoder

IMAGE_SIZE = 128
TEXT_DIM = 384
TOKEN_LEN = 24
TIMESTEPS = 1000

DEFAULT_PROMPTS = [
    "a dog running through green grass",
    "a person in a red shirt on the street",
    "a child playing outside with a ball",
    "a brown dog jumping into water",
    "two men standing on a rocky mountain",
    "a girl in a pink dress smiling",
]


def parse_args():
    p = argparse.ArgumentParser(description="Generate 128x128 images from text prompts.")
    p.add_argument("--weights", default="t2i_weights_v2.h5", help="Path to local weights (.h5)")
    p.add_argument(
        "--hf-repo",
        default="geeteshcodes/t2i-coco-128-clipL",
        help="Hugging Face repo to pull weights from if local file is missing",
    )
    p.add_argument("--prompts", nargs="*", default=None, help="Custom text prompts")
    p.add_argument("--scales", nargs="*", type=float, default=[3.0, 5.0, 7.5], help="Guidance scales")
    p.add_argument("--steps", type=int, default=50, help="DDIM reverse sampling steps (20-50 recommended)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--base-filters", type=int, default=32, help="Base U-Net filter channels (32 or 48)")
    p.add_argument("--out", default="generated_grid.png", help="Output PNG grid file path")
    return p.parse_args()


def ensure_weights(weights_path: str, hf_repo: str) -> str:
    """Checks if weights file exists locally; if not, downloads from Hugging Face Hub."""
    if os.path.exists(weights_path):
        return weights_path

    print(f"[Model] Local weights '{weights_path}' not found.")
    print(f"[Model] Downloading from Hugging Face Hub repo '{hf_repo}'...")
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(repo_id=hf_repo, filename=os.path.basename(weights_path))
        print(f"[Model] Successfully downloaded weights to: {downloaded}")
        return downloaded
    except Exception as e:
        raise FileNotFoundError(
            f"Could not find or download model weights '{weights_path}' from '{hf_repo}': {e}\n"
            f"Please download 't2i_weights_v2.h5' manually from https://huggingface.co/{hf_repo}"
        )


def build_model(weights_path: str, base_filters: int):
    model = TextToImageUNet(base_filters=base_filters, text_dim=TEXT_DIM)
    dummy = [
        tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3], tf.float32),
        tf.zeros([1], tf.int32),
        tf.zeros([1, TOKEN_LEN, TEXT_DIM + 1], tf.float32),
    ]
    model(dummy, training=False)
    model.load_weights(weights_path)
    return model


def to_uint8(images):
    return ((images.numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def main():
    args = parse_args()
    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS

    weights_file = ensure_weights(args.weights, args.hf_repo)
    model = build_model(weights_file, args.base_filters)
    print(f"[Model] Loaded weights from {weights_file}: {model.count_params():,} parameters")

    print(f"[Text] Encoding {len(prompts)} prompt(s) via TextEncoder...")
    encoder = TextEncoder()
    embeddings = tf.convert_to_tensor(encoder.encode(prompts), tf.float32)

    diffusion = GaussianDiffusion(timesteps=TIMESTEPS)
    shape = [len(prompts), IMAGE_SIZE, IMAGE_SIZE, 3]

    columns = []
    for scale in args.scales:
        print(f"[Sampling] Running DDIM ({args.steps} steps) with guidance scale {scale}...")
        tf.random.set_seed(args.seed)
        images = diffusion.sample_text_to_image(
            model, embeddings, shape, guidance_scale=scale, ddim_steps=args.steps
        )
        columns.append(to_uint8(images))

    rows = [np.concatenate([col[i] for col in columns], axis=1) for i in range(len(prompts))]
    grid = np.concatenate(rows, axis=0)
    tf.io.write_file(args.out, tf.io.encode_png(grid))

    print(f"\n[Done] Saved generated grid to: {args.out}")
    print("Prompts (top to bottom):")
    for i, prompt in enumerate(prompts):
        print(f"  {i + 1}. \"{prompt}\"")
    print("Guidance scales (left to right): " + ", ".join(f"scale={s}" for s in args.scales))


if __name__ == "__main__":
    main()
