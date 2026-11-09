"""
Kaggle TPU / Multi-GPU Training for Text-to-Image Diffusion U-Net (v2).
Built for quota-constrained sessions (TPU ~20h/week, GPU ~30h/week) with automatic checkpointing,
min-SNR-gamma loss weighting, EMA weights tracking, and seamless Hugging Face Hub persistence.

Supported Datasets:
- MS-COCO 2017 (captions_train2017.json + train2017 images)
- Flickr30k (captions.txt + flickr30k_images)
- Flickr8k (captions.txt / Flickr8k.token.txt + Images)
- Custom image-caption dataset via T2I_CAPTIONS and T2I_IMAGES env vars
"""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # Targets Keras 2 (tf-keras)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import glob
import inspect
import math
import random
import re
import shutil
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import tensorflow as tf

try:
    _kv = str(getattr(tf.keras, "__version__", "2"))
except ImportError:
    raise SystemExit("tf-keras is missing. Run:  pip install tf-keras  (same minor version as TensorFlow)")
if _kv.startswith("3"):
    raise SystemExit("Keras 3 detected. Run:  pip install tf-keras  and rerun with TF_USE_LEGACY_KERAS=1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from architecture import TextToImageUNet, TOKEN_LEN
from diffusion import GaussianDiffusion
from dataset import parse_captions_file


# ------------------------------- Configuration -------------------------------
def _env(name, default, cast=str):
    return cast(os.environ.get(name, default))


SMOKE = os.environ.get("T2I_SMOKE") == "1"
FAKE_DATA = os.environ.get("T2I_FAKE_DATA") == "1"
IMAGE_SIZE = _env("T2I_IMAGE_SIZE", 128, int)
BASE_FILTERS = _env("T2I_BASE_FILTERS", 32, int)
TIMESTEPS = 1000
TEXT_DIM = 384
PER_REPLICA_BATCH = _env("T2I_PER_REPLICA_BATCH", 32, int)  # global batch = this x replicas
TARGET_PASSES = _env("T2I_TARGET_PASSES", 100, float)       # passes over training pairs
TEXT_DROPOUT = 0.15
WARMUP_SAMPLES = 40_000
MIN_LR = 1e-5
GRAD_CLIP = 1.0
EMA_DECAY = 0.999
MIN_SNR_GAMMA = 5.0

# Session limits (Kaggle hard-stops at ~12h GPU / ~9h TPU): leave time to save + upload
SESSION_HOURS = {"TPU": 8.4, "GPU": 11.0, "CPU": 0.2}
CKPT_MIN, UPLOAD_MIN, PREVIEW_MIN, VAL_MIN, SNAPSHOT_HOURS = 10, 30, 30, 20, 3
LOG_SEC = _env("T2I_LOG_SEC", 60, float)
CHUNK_STEPS = {"TPU": 50, "GPU": 20, "CPU": 2}  # steps per compiled call
VAL_BATCHES, VAL_BATCH = 32, 32
MAX_ROLLBACKS = 3

# Hugging Face Hub persistence: set HF_REPO (e.g. 'geeteshcodes/t2i-coco-128-clipL') and HF_TOKEN secret
HF_REPO = os.environ.get("HF_REPO", "")

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
RUN = WORK / "t2i_run"
CKPT_DIR, SAMPLES_DIR, LOG_DIR, CACHE_DIR = RUN / "ckpt", RUN / "samples", RUN / "logs", RUN / "cache"
STAGE_DIR, HF_DL = RUN / "stage", RUN / "hf_dl"
WEIGHTS_OUT = RUN / "t2i_weights_v2.h5"

for d in (CKPT_DIR, SAMPLES_DIR, LOG_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

TEST_PROMPTS = [
    "a dog running through green grass",
    "a person in a red shirt on the street",
    "a child playing outside with a ball",
    "a brown dog jumping into water",
]
STOP = {"flag": False}


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_DIR / "train.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _on_signal(signum, frame):
    STOP["flag"] = True
    log(f"Signal {signum}: will save and exit after the current chunk.")


for _n in ("SIGINT", "SIGTERM"):
    try:
        signal.signal(getattr(signal, _n), _on_signal)
    except (ValueError, OSError, AttributeError):
        pass


# ------------------------------- Persistence -------------------------------
class Persist:
    """Hugging Face Hub sync so checkpoints and weights survive Kaggle session timeouts."""
    def __init__(self):
        self.repo, self.token, self.api = HF_REPO, None, None
        self.thread = None
        if not self.repo:
            return
        try:
            from kaggle_secrets import UserSecretsClient
            self.token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            self.token = os.environ.get("HF_TOKEN")
        if not self.token:
            log("HF_REPO set but no HF_TOKEN found; HF sync disabled.")
            return
        try:
            from huggingface_hub import HfApi
            self.api = HfApi(token=self.token)
            self.api.create_repo(self.repo, private=True, exist_ok=True)
            log(f"Hugging Face sync enabled: {self.repo}")
        except Exception as e:
            log(f"HF init failed ({e}); sync disabled.")
            self.api = None

    @property
    def enabled(self):
        return self.api is not None

    def download(self, patterns, local_dir):
        if not self.enabled:
            return False
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=self.repo, allow_patterns=patterns, local_dir=str(local_dir), token=self.token)
            return True
        except Exception as e:
            log(f"HF download skipped ({type(e).__name__}).")
            return False

    def upload_async(self, folder, path_in_repo, delete_patterns=None):
        if not self.enabled or (self.thread and self.thread.is_alive()):
            return

        def work():
            try:
                self.api.upload_folder(
                    repo_id=self.repo,
                    folder_path=str(folder),
                    path_in_repo=path_in_repo,
                    delete_patterns=delete_patterns,
                    commit_message=f"{path_in_repo} {time.strftime('%Y-%m-%d %H:%M')}",
                )
                log(f"Uploaded {path_in_repo} to Hugging Face.")
            except Exception as e:
                log(f"HF upload failed (training continues): {e}")

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def wait(self, timeout=900):
        if self.thread:
            self.thread.join(timeout)


def list_ckpts(d):
    out = []
    for f in Path(d).glob("ckpt-*.index"):
        m = re.match(r"ckpt-(\d+)\.index$", f.name)
        if m:
            out.append((int(m.group(1)), str(f)[: -len(".index")]))
    return sorted(out, reverse=True)


def prune(d, keep=2):
    for _, prefix in list_ckpts(d)[keep:]:
        for f in glob.glob(prefix + ".*"):
            try:
                os.remove(f)
            except OSError:
                pass


def candidate_ckpt_dirs():
    dirs = [CKPT_DIR, HF_DL / "ckpt"]
    for pat in ("/kaggle/input/*/t2i_run/ckpt", "/kaggle/input/*/ckpt", "/kaggle/input/*/*/t2i_run/ckpt"):
        dirs += [Path(p) for p in glob.glob(pat)]
    return dirs


# ------------------------------- Data Discovery -------------------------------
def find_dataset():
    """
    Locates caption annotations file and image directory.
    Checks environment variables (T2I_CAPTIONS, T2I_IMAGES), MS-COCO 2017, and Flickr.
    """
    if os.environ.get("T2I_CAPTIONS") and os.environ.get("T2I_IMAGES"):
        return Path(os.environ["T2I_CAPTIONS"]), Path(os.environ["T2I_IMAGES"])

    # 1. COCO 2017 detection
    coco_cap_pats = [
        "/kaggle/input/**/captions_train2017.json",
        "/kaggle/input/**/annotations/captions_train2017.json",
        "data/**/captions_train2017.json",
    ]
    for pat in coco_cap_pats:
        hits = glob.glob(pat, recursive=True)
        if hits:
            cap_file = Path(hits[0])
            img_pats = [
                str(cap_file.parent.parent / "train2017"),
                "/kaggle/input/**/train2017",
                "data/**/train2017",
            ]
            for ipat in img_pats:
                ihits = glob.glob(ipat, recursive=True)
                if ihits and Path(ihits[0]).is_dir():
                    return cap_file, Path(ihits[0])

    # 2. Flickr detection
    flickr_pats = [
        "/kaggle/input/*/captions.txt",
        "/kaggle/input/*/*/captions.txt",
        "data/flickr30k/captions.txt",
        "data/flickr8k/captions.txt",
        "data/captions.txt",
        "/kaggle/input/*/Flickr8k.token.txt",
    ]
    for p in flickr_pats:
        hits = glob.glob(p)
        if hits:
            cap_file = Path(hits[0])
            img_dir = cap_file.parent / "Images"
            if not img_dir.exists():
                img_dir = cap_file.parent / "flickr30k_images"
            if not img_dir.exists():
                img_dir = cap_file.parent
            return cap_file, img_dir

    raise FileNotFoundError(
        "No dataset found! Provide T2I_CAPTIONS and T2I_IMAGES environment variables, "
        "or add MS-COCO 2017 ('awsaf49/coco-2017-dataset') or Flickr to /kaggle/input."
    )


def _load_img(path):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    l, t = (w - s) // 2, (h - s) // 2
    im = im.crop((l, t, l + s, t + s)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    return np.asarray(im, dtype=np.uint8)


def build_arrays(persist):
    """Loads or creates precomputed image and token embedding arrays."""
    files = [CACHE_DIR / n for n in ("images.npy", "pair_img.npy", "embs.npy")]
    if FAKE_DATA:
        rng = np.random.default_rng(0)
        n_img = 48
        images = rng.integers(0, 255, (n_img, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        pair_img = np.repeat(np.arange(n_img, dtype=np.int32), 3)
        embs = rng.standard_normal((len(pair_img), TOKEN_LEN, TEXT_DIM + 1)).astype(np.float16)
        embs[..., -1] = (np.arange(TOKEN_LEN) < 8).astype(np.float16)
        embs[..., :-1] *= embs[..., -1:]
        return images, pair_img, embs, embs[:4].astype(np.float32)

    from text_encoder import TextEncoder
    enc = TextEncoder()
    if enc.backend != "sentence_transformers":
        raise RuntimeError("pip install sentence-transformers (required for token-level text embeddings).")
    test_embs = enc.encode(TEST_PROMPTS)

    if not all(f.exists() for f in files) and persist.enabled:
        persist.download(["cache/*"], RUN)

    if all(f.exists() for f in files):
        log("Loading cached arrays from disk.")
        images, pair_img, embs = (np.load(f, mmap_mode=None) for f in files)
        if embs.shape[1:] == (TOKEN_LEN, TEXT_DIM + 1) and images.shape[1] == IMAGE_SIZE:
            return images, pair_img, embs, test_embs
        log("Cache shape mismatch; rebuilding cache arrays...")

    cap_file, img_dir = find_dataset()
    log(f"Building arrays from annotations: {cap_file} and images: {img_dir}...")
    caps = parse_captions_file(cap_file)

    names = [n for n in caps if (img_dir / n).exists() or (img_dir / f"{n}.jpg").exists()]
    paths = [img_dir / n if (img_dir / n).exists() else img_dir / f"{n}.jpg" for n in names]

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as ex:
        images = np.stack(list(ex.map(_load_img, paths)))

    pair_img, texts = [], []
    for i, n in enumerate(names):
        for c in caps[n]:
            if c.strip():
                pair_img.append(i)
                texts.append(c.strip())

    pair_img = np.array(pair_img, dtype=np.int32)
    embs = np.empty((len(texts), TOKEN_LEN, TEXT_DIM + 1), dtype=np.float16)

    for k in range(0, len(texts), 256):
        embs[k : k + 256] = enc.encode(texts[k : k + 256], fp16=True)
        if (k // 256) % 25 == 0:
            log(f"  Tokenizing & encoding text: {k}/{len(texts)}")

    for f, a in zip(files, (images, pair_img, embs)):
        np.save(f, a)

    log(f"Arrays cached successfully: {len(images)} images, {len(texts)} image-caption pairs.")
    persist.upload_async(CACHE_DIR, "cache")
    return images, pair_img, embs, test_embs


def make_ds(images, pair_img, embs, lo, hi, batch, training):
    """Memory-efficient tf.data pipeline using generator streaming with batch prefetching."""
    n = hi - lo

    def gen():
        rng = np.random.default_rng()
        while True:
            order = (rng.permutation(n) + lo) if training else np.arange(lo, hi)
            for i in order:
                yield images[pair_img[i]], embs[i]
            if not training:
                return

    sig = (
        tf.TensorSpec((IMAGE_SIZE, IMAGE_SIZE, 3), tf.uint8),
        tf.TensorSpec(tuple(embs.shape[1:]), tf.float16),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=sig)

    def load(img, emb):
        img = tf.cast(img, tf.float32) / 127.5 - 1.0
        emb = tf.cast(emb, tf.float32)
        if training:
            img = tf.image.random_flip_left_right(img)
            emb = tf.where(tf.random.uniform([]) < TEXT_DROPOUT, tf.zeros_like(emb), emb)  # CFG null conditioning
        return img, emb

    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


# ------------------------------- Distributed Strategy -------------------------------
def make_strategy():
    fake_cpus = _env("T2I_FAKE_CPUS", 0, int)
    if fake_cpus:
        cpu = tf.config.list_physical_devices("CPU")[0]
        tf.config.set_logical_device_configuration(cpu, [tf.config.LogicalDeviceConfiguration()] * fake_cpus)
        return tf.distribute.MirroredStrategy([f"CPU:{i}" for i in range(fake_cpus)]), "CPU"
    try:
        try:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        except ValueError:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        return tf.distribute.TPUStrategy(resolver), "TPU"
    except Exception as e:
        log(f"No TPU detected ({type(e).__name__}); checking for GPUs...")

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass
    if len(gpus) > 1:
        return tf.distribute.MirroredStrategy(), "GPU"
    return tf.distribute.get_strategy(), ("GPU" if gpus else "CPU")


# ------------------------------- Main Training Loop -------------------------------
def main():
    strategy, kind = make_strategy()
    R = strategy.num_replicas_in_sync
    GLOBAL_BATCH = PER_REPLICA_BATCH * R
    PEAK_LR = min(1e-4 * math.sqrt(GLOBAL_BATCH / 8.0), 3e-4)
    session_s = _env("T2I_SESSION_HOURS", SESSION_HOURS[kind], float) * 3600
    CHUNK = CHUNK_STEPS[kind]
    log("=" * 70)
    log(
        f"Device: {kind}, Replicas: {R}, Global Batch: {GLOBAL_BATCH}, Peak LR: {PEAK_LR:.2e}, "
        f"Session Limit: {session_s / 3600:.1f} h"
    )

    persist = Persist()
    if persist.enabled:
        persist.download(["ckpt/*", "t2i_weights_v2.h5"], HF_DL)

    images, pair_img, embs, test_embs_np = build_arrays(persist)
    n_pairs = len(pair_img)
    n_train = int(n_pairs * 0.9)
    TOTAL_SAMPLES = int(TARGET_PASSES * n_train)
    log(f"Data: {len(images)} unique images, {n_pairs} pairs ({n_train} train); Target: {TOTAL_SAMPLES:,} samples")

    train_ds = make_ds(images, pair_img, embs, 0, n_train, GLOBAL_BATCH, True)
    val_ds = make_ds(images, pair_img, embs, n_train, n_pairs, min(VAL_BATCH, max(n_pairs - n_train, 1)), False)
    dist_ds = strategy.experimental_distribute_dataset(train_ds)
    it = iter(dist_ds)

    diffusion = GaussianDiffusion(timesteps=TIMESTEPS, schedule="cosine")
    ab_const = diffusion.alphas_cumprod
    dummy = [
        tf.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3)),
        tf.zeros((1,), tf.int32),
        tf.zeros((1, TOKEN_LEN, TEXT_DIM + 1)),
    ]

    with strategy.scope():
        network = TextToImageUNet(base_filters=BASE_FILTERS, text_dim=TEXT_DIM)

        @tf.function
        def build_net():
            strategy.run(lambda: network(dummy))

        build_net()
        net_vars = network.variables
        train_vars = network.trainable_variables
        ema_vars = [
            tf.Variable(tf.zeros(v.shape, v.dtype), trainable=False, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
            for v in net_vars
        ]
        optimizer = tf.keras.optimizers.Adam(learning_rate=PEAK_LR)
        if hasattr(optimizer, "build"):
            optimizer.build(train_vars)
        elif hasattr(optimizer, "_create_all_weights"):
            optimizer._create_all_weights(train_vars)

    log(f"Model parameters: {network.count_params():,}")

    sig = inspect.signature(optimizer.apply_gradients).parameters
    NO_AGG = (
        {"skip_gradients_aggregation": True}
        if "skip_gradients_aggregation" in sig
        else {"experimental_aggregate_gradients": False}
        if "experimental_aggregate_gradients" in sig
        else {}
    )

    step_var = tf.Variable(0, dtype=tf.int64, trainable=False)
    samples_var = tf.Variable(0, dtype=tf.int64, trainable=False)
    ckpt = tf.train.Checkpoint(network=network, ema=ema_vars, optimizer=optimizer, step=step_var, samples=samples_var)
    CK_OPTS = tf.train.CheckpointOptions(experimental_io_device="/job:localhost") if kind == "TPU" else None

    # Step function with Min-SNR-gamma weighting
    def step_fn(x, emb, seed_step, decay):
        rid = tf.cast(tf.distribute.get_replica_context().replica_id_in_sync_group, tf.int32)
        seed = tf.stack([seed_step, rid])
        b = tf.shape(x)[0]
        t = tf.random.stateless_uniform([b], seed=seed, minval=0, maxval=TIMESTEPS, dtype=tf.int32)
        noise = tf.random.stateless_normal(tf.shape(x), seed=seed + 7919)
        x_noisy = diffusion.q_sample(x, t, noise)

        with tf.GradientTape() as tape:
            pred = network([x_noisy, t, emb], training=True)
            mse = tf.reduce_mean(tf.square(noise - pred), axis=[1, 2, 3])
            ab = tf.gather(ab_const, t)
            snr = ab / tf.maximum(1.0 - ab, 1e-8)
            w = tf.minimum(snr, MIN_SNR_GAMMA) / tf.maximum(snr, 1e-8)  # Min-SNR-gamma weighting
            loss = tf.nn.compute_average_loss(w * mse, global_batch_size=GLOBAL_BATCH)

        grads = tape.gradient(loss, train_vars)
        grads = tf.distribute.get_replica_context().all_reduce(tf.distribute.ReduceOp.SUM, grads)
        gnorm = tf.linalg.global_norm(grads)
        ok = tf.math.is_finite(gnorm)
        grads, _ = tf.clip_by_global_norm(grads, GRAD_CLIP, use_norm=gnorm)
        grads = [tf.where(ok, g, tf.zeros_like(g)) for g in grads]
        optimizer.apply_gradients(zip(grads, train_vars), **NO_AGG)

        for e, v in zip(ema_vars, net_vars):
            e.assign(decay * e + (1.0 - decay) * v)

        return (
            loss,
            tf.nn.compute_average_loss(mse, global_batch_size=GLOBAL_BATCH),
            gnorm,
            1.0 - tf.cast(ok, tf.float32),
        )

    @tf.function
    def train_chunk(iterator, seed0, decay):
        ls, ms, gs, bs = tf.zeros([]), tf.zeros([]), tf.zeros([]), tf.zeros([])
        for i in tf.range(CHUNK):
            x, emb = next(iterator)
            l, m, g, bad = strategy.run(step_fn, args=(x, emb, seed0 + i, decay))
            ls += strategy.reduce(tf.distribute.ReduceOp.SUM, l, axis=None)
            ms += strategy.reduce(tf.distribute.ReduceOp.SUM, m, axis=None)
            gs += strategy.reduce(tf.distribute.ReduceOp.MEAN, g, axis=None)
            bs += strategy.reduce(tf.distribute.ReduceOp.MEAN, bad, axis=None)
        return ls, ms, gs, bs

    @tf.function
    def ema_copy():
        def f():
            for e, v in zip(ema_vars, net_vars):
                e.assign(v)

        strategy.run(f)

    # Host-side model for validation / previews / export (EMA weights)
    dev = "/CPU:0" if kind == "TPU" else None

    class _Ctx:
        def __enter__(s):
            s.c = tf.device(dev) if dev else None
            if s.c:
                s.c.__enter__()

        def __exit__(s, *a):
            if s.c:
                s.c.__exit__(*a)

    with _Ctx():
        host_net = TextToImageUNet(base_filters=BASE_FILTERS, text_dim=TEXT_DIM)
        host_net(dummy)

    def load_ema_into_host():
        for hv, ev in zip(host_net.variables, ema_vars):
            hv.assign(ev.numpy())

    @tf.function
    def val_step(x, emb, i):
        b = tf.shape(x)[0]
        stride = tf.maximum(TIMESTEPS // b, 1)
        t = (tf.range(b) * TIMESTEPS) // b + (i * 37) % stride
        noise = tf.random.stateless_normal(tf.shape(x), seed=tf.stack([i, tf.constant(1234, tf.int32)]))
        pred = host_net([diffusion.q_sample(x, t, noise), t, emb], training=False)
        return tf.reduce_mean(tf.square(noise - pred))

    def validate():
        try:
            with _Ctx():
                load_ema_into_host()
                vals = [float(val_step(x, e, tf.constant(i, tf.int32))) for i, (x, e) in enumerate(val_ds.take(VAL_BATCHES))]
            return float(np.mean(vals))
        except Exception as e:
            log(f"Validation failed (ignored): {e}")
            return float("nan")

    def export_ema(path):
        with _Ctx():
            load_ema_into_host()
            tmp = Path(str(path) + ".tmp.h5")
            host_net.save_weights(str(tmp))
            os.replace(tmp, path)

    def make_preview(step):
        try:
            with _Ctx():
                load_ema_into_host()
                te = tf.convert_to_tensor(test_embs_np, tf.float32)
                s = diffusion.sample_text_to_image(
                    host_net, te, [te.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3], guidance_scale=3.0, ddim_steps=30
                )
            im = ((s.numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            path = SAMPLES_DIR / f"t2i_step_{step:08d}.png"
            tf.io.write_file(str(path), tf.io.encode_png(np.concatenate(list(im), axis=1)))
            log(f"Preview saved: {path.name}")
        except Exception as e:
            log(f"Preview failed (ignored): {e}")

    def save_ckpt(step, samples):
        step_var.assign(step)
        samples_var.assign(samples)
        ckpt.write(str(CKPT_DIR / f"ckpt-{step:09d}"), options=CK_OPTS)
        prune(CKPT_DIR, keep=2)

    def restore_best():
        cands = []
        for d in candidate_ckpt_dirs():
            cands += [(s, p) for s, p in list_ckpts(d)]
        for s, p in sorted(set(cands), reverse=True):
            try:
                ckpt.restore(p, options=CK_OPTS).expect_partial()
                log(f"Restored checkpoint {p} (step {int(step_var.numpy())})")
                return True
            except Exception as e:
                log(f"Checkpoint {p} unreadable ({type(e).__name__}); trying an older one...")
        return False

    def stage_and_upload(step):
        if not persist.enabled:
            return
        lc = list_ckpts(CKPT_DIR)
        if not lc:
            return
        shutil.rmtree(STAGE_DIR, ignore_errors=True)
        (STAGE_DIR / "ckpt").mkdir(parents=True)
        for f in glob.glob(lc[0][1] + ".*"):
            shutil.copy(f, STAGE_DIR / "ckpt")
        if WEIGHTS_OUT.exists():
            shutil.copy(WEIGHTS_OUT, STAGE_DIR / WEIGHTS_OUT.name)
        persist.wait(60)
        persist.upload_async(STAGE_DIR, ".", delete_patterns=["ckpt/ckpt-*"])

    if restore_best():
        step, samples = int(step_var.numpy()), int(samples_var.numpy())
    else:
        ema_copy()
        step = samples = 0
        log("Fresh run from random initialization.")

    log(f"Progress: step {step}, {samples:,}/{TOTAL_SAMPLES:,} samples ({samples / max(n_train, 1):.1f} passes)")

    def lr_at(s):
        if s < WARMUP_SAMPLES:
            return PEAK_LR * (s + 1) / WARMUP_SAMPLES
        p = min(1.0, (s - WARMUP_SAMPLES) / max(TOTAL_SAMPLES - WARMUP_SAMPLES, 1))
        return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1.0 + math.cos(math.pi * p))

    seed_base = random.randint(0, 2 ** 30)
    t_start = last_ckpt = last_up = last_prev = last_val = last_snap = last_log = time.time()
    win_l = win_m = win_g = 0.0
    win_n, win_s0, rollbacks, bad_chunks = 0, samples, 0, 0
    reason, code = "unknown", 0

    while True:
        now = time.time()
        if STOP["flag"]:
            reason = "stop signal"
            break
        if now - t_start >= session_s:
            reason = "session time limit (resume next session)"
            break
        if samples >= TOTAL_SAMPLES:
            reason = "TARGET REACHED"
            break

        tf.keras.backend.set_value(optimizer.learning_rate, lr_at(samples))
        decay = min(EMA_DECAY, (1.0 + step) / (10.0 + step))
        ls, ms, gs, bs = train_chunk(
            it, tf.constant((seed_base + step) % (2 ** 31 - 1), tf.int32), tf.constant(decay, tf.float32)
        )
        l, m, g, bad = float(ls) / CHUNK, float(ms) / CHUNK, float(gs) / CHUNK, float(bs) / CHUNK
        step += CHUNK
        samples += CHUNK * GLOBAL_BATCH
        win_l += l
        win_m += m
        win_g += g
        win_n += 1

        if bad > 0.5 or not np.isfinite(l):
            bad_chunks += 1
            if bad_chunks >= 3:
                rollbacks += 1
                log(f"Divergence detected. Rollback {rollbacks}/{MAX_ROLLBACKS}.")
                if rollbacks > MAX_ROLLBACKS or not restore_best():
                    reason, code = "diverged", 3
                    break
                step, samples = int(step_var.numpy()), int(samples_var.numpy())
                bad_chunks = 0
        else:
            bad_chunks = 0

        now = time.time()
        if now - last_log >= LOG_SEC:
            sps = (samples - win_s0) / max(now - last_log, 1e-6)
            eta_h = (TOTAL_SAMPLES - samples) / max(sps, 1e-6) / 3600
            log(
                f"pass {samples / n_train:.1f}/{TARGET_PASSES:.0f} step {step} loss {win_l / win_n:.4f} "
                f"mse {win_m / win_n:.4f} gnorm {win_g / win_n:.2f} lr {lr_at(samples):.2e} "
                f"{sps:.0f} samples/s | session {(now - t_start) / 3600:.2f}h | ~{eta_h:.1f}h to target"
            )
            win_l = win_m = win_g = 0.0
            win_n = 0
            win_s0 = samples
            last_log = now

        if now - last_ckpt >= CKPT_MIN * 60:
            save_ckpt(step, samples)
            last_ckpt = now

        if now - last_val >= VAL_MIN * 60:
            log(f"step {step} EMA val_mse (fixed t/noise): {validate():.4f}")
            last_val = time.time()

        if now - last_prev >= PREVIEW_MIN * 60:
            make_preview(step)
            last_prev = time.time()

        if now - last_up >= UPLOAD_MIN * 60:
            try:
                export_ema(WEIGHTS_OUT)
            except Exception as e:
                log(f"EMA export failed (ignored): {e}")
            stage_and_upload(step)
            last_up = time.time()

        if now - last_snap >= SNAPSHOT_HOURS * 3600:
            try:
                export_ema(RUN / f"ema_step_{step:08d}.h5")
            except Exception as e:
                log(f"Snapshot failed (ignored): {e}")
            last_snap = time.time()

    # Wrap up session
    log(f"Stopping: {reason} at step {step}, {samples:,} samples ({samples / n_train:.1f} passes)")
    if code != 3:
        save_ckpt(step, samples)
        try:
            export_ema(WEIGHTS_OUT)
        except Exception as e:
            log(f"Final EMA export failed: {e}")
        make_preview(step)
        stage_and_upload(step)
        persist.wait(900)

    log(
        "Session finished. Start notebook again next session to continue."
        if code == 0 and "TARGET" not in reason
        else f"Done: {reason}"
    )
    return code


if __name__ == "__main__":
    if SMOKE:
        os.environ.setdefault("T2I_SESSION_HOURS", "0.004")
    try:
        rc = main()
    except SystemExit:
        raise
    except BaseException:
        log("CRASH:\n" + traceback.format_exc())
        rc = 1
    sys.exit(rc)
