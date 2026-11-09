"""
Universal Dataset Loader & Text Embedding Precomputation.
Supports:
- MS-COCO 2017 annotations (captions_train2017.json, captions_val2017.json)
- Flickr30k ('|' pipe separated format)
- Flickr8k (Flickr8k.token.txt or CSV image,caption format)
- Custom directories with images and paired captions
- High-performance tf.data input pipeline with Text Dropout for Classifier-Free Guidance (CFG)
"""

import csv
import json
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from text_encoder import TextEncoder, TEXT_EMBED_DIM, TOKEN_LEN


def parse_captions_file(captions_path: Path):
    """
    Parses captions from multiple formats:
    - COCO format JSON (e.g., captions_train2017.json)
    - Flickr30k format (| separated: image_name|caption_id|caption)
    - Flickr8k format (tab-separated: image_id#0\tcaption)
    - CSV format (image,caption)

    Returns:
        dict mapping image_filename -> list of captions
    """
    captions_path = Path(captions_path)
    image_to_captions = {}

    # 1. COCO JSON format
    if captions_path.suffix.lower() == ".json":
        print(f"[Dataset] Parsing COCO format JSON annotations from: {captions_path}")
        with open(captions_path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)

        id_to_file = {img["id"]: img["file_name"] for img in data.get("images", [])}
        for ann in data.get("annotations", []):
            img_id = ann.get("image_id")
            filename = id_to_file.get(img_id)
            caption = ann.get("caption", "").strip()
            if filename and caption:
                image_to_captions.setdefault(filename, []).append(caption)
        return image_to_captions

    # 2. Text / CSV / TSV formats
    with open(captions_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()
        f.seek(0)

        # Flickr30k pipe-delimited format: image_name|comment_number|comment
        if "|" in first_line:
            for line in f:
                parts = line.rstrip("\n").split("|", 2)
                if len(parts) == 3 and parts[1].strip().isdigit() and parts[2].strip():
                    img_name = parts[0].strip()
                    caption = parts[2].strip()
                    image_to_captions.setdefault(img_name, []).append(caption)

        # Flickr8k token format: image.jpg#0 \t caption
        elif "\t" in first_line:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) == 2 and parts[1].strip():
                    img_name = parts[0].split("#")[0].strip()
                    caption = parts[1].strip()
                    image_to_captions.setdefault(img_name, []).append(caption)

        # Standard CSV format: image,caption
        else:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2 and not row[0].lower().startswith("image"):
                    img_name = row[0].strip()
                    caption = ",".join(row[1:]).strip()
                    if img_name and caption:
                        image_to_captions.setdefault(img_name, []).append(caption)

    return image_to_captions


def find_dataset_files(search_dirs=None):
    """
    Searches standard directories for MS-COCO or Flickr datasets.
    Returns (captions_file, images_dir).
    """
    coco_caption_candidates = [
        Path("/kaggle/input/coco-2017-dataset/coco2017/annotations/captions_train2017.json"),
        Path("/kaggle/input/coco-2017-dataset/annotations/captions_train2017.json"),
        Path("/kaggle/input/annotations/captions_train2017.json"),
        Path("data/annotations/captions_train2017.json"),
        Path("data/coco/annotations/captions_train2017.json"),
    ]
    coco_image_candidates = [
        Path("/kaggle/input/coco-2017-dataset/coco2017/train2017"),
        Path("/kaggle/input/coco-2017-dataset/train2017"),
        Path("/kaggle/input/train2017"),
        Path("data/train2017"),
        Path("data/coco/train2017"),
    ]

    for cap_path in coco_caption_candidates:
        if cap_path.exists():
            for img_path in coco_image_candidates:
                if img_path.exists():
                    return cap_path, img_path

    # Flickr candidates
    flickr_caption_candidates = [
        Path("/kaggle/input/flickr30k/captions.txt"),
        Path("/kaggle/input/flickr8k/captions.txt"),
        Path("data/flickr30k/captions.txt"),
        Path("data/flickr8k/captions.txt"),
        Path("data/captions.txt"),
    ]
    for cap_path in flickr_caption_candidates:
        if cap_path.exists():
            img_dir = cap_path.parent / "Images"
            if not img_dir.exists():
                img_dir = cap_path.parent / "flickr30k_images"
            if not img_dir.exists():
                img_dir = cap_path.parent
            return cap_path, img_dir

    return None, None


def create_tf_dataset(
    image_paths,
    embeddings,
    batch_size=32,
    image_size=(128, 128),
    text_dropout_prob=0.15,
    is_training=True,
):
    """
    Creates an optimized tf.data.Dataset pipeline.
    Randomly drops text embeddings with probability `text_dropout_prob` for CFG training.
    """
    emb_shape = list(embeddings.shape[1:])
    image_paths = np.array(image_paths)

    def load_and_preprocess(path, emb):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        side = tf.minimum(tf.shape(img)[0], tf.shape(img)[1])
        img = tf.image.resize_with_crop_or_pad(img, side, side)
        img = tf.image.resize(img, image_size)
        if is_training:
            img = tf.image.random_flip_left_right(img)
        img = tf.cast(img, tf.float32) / 127.5 - 1.0  # Normalize to [-1, 1]
        img.set_shape([image_size[0], image_size[1], 3])

        emb = tf.cast(emb, tf.float32)
        emb.set_shape(emb_shape)

        # Text dropout for Classifier-Free Guidance
        if is_training and text_dropout_prob > 0:
            cond = tf.random.uniform([]) < text_dropout_prob
            emb = tf.where(cond, tf.zeros_like(emb), emb)

        return img, emb

    ds = tf.data.Dataset.from_tensor_slices((image_paths, embeddings))
    if is_training:
        ds = ds.shuffle(buffer_size=min(len(image_paths), 10000), reshuffle_each_iteration=True)

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    if is_training:
        ds = ds.repeat()

    return ds
