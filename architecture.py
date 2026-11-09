"""
Text-to-Image Diffusion U-Net v2.
- Token-level cross-attention (every text token is a key/value) at 32x32 and 16x16
- Self-attention at 16x16 (optional at 32x32 via attn32=True)
- 4 resolution levels (128/64/32/16), channels (f, 2f, 4f, 8f); ~6M params at base_filters=32
- FiLM + sinusoidal time embedding + SE channel attention kept from v1
Text input format (packed): (B, TOKEN_LEN, text_dim + 1) = token embeddings + 1 mask channel.
All-zeros (or a rank-2 tensor) means "no text" = classifier-free-guidance null.
"""

import math
import tensorflow as tf
from tensorflow.keras import layers

TOKEN_LEN = 24


def get_sinusoidal_time_embedding(timesteps, embedding_dim=128):
    """Computes sinusoidal positional embeddings for diffusion timesteps t."""
    timesteps = tf.cast(timesteps, tf.float32)
    half_dim = embedding_dim // 2
    emb = math.log(10000.0) / (half_dim - 1)
    emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
    return emb


class TimeEmbedding(layers.Layer):
    """MLP to project sinusoidal embeddings into a rich conditioning vector."""
    def __init__(self, dim=128, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.dense1 = layers.Dense(dim * 2, activation="swish")
        self.dense2 = layers.Dense(dim, activation="swish")

    def call(self, t):
        emb = get_sinusoidal_time_embedding(t, self.dim)
        emb = self.dense1(emb)
        emb = self.dense2(emb)
        return emb


class TextProjection(layers.Layer):
    """Projects dense text embeddings (e.g. 384-dim) into conditioning dimension."""
    def __init__(self, in_dim=384, out_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = layers.Dense(out_dim * 2, activation="swish")
        self.dense2 = layers.Dense(out_dim, activation="swish")

    def call(self, text_emb):
        x = self.dense1(text_emb)
        x = self.dense2(x)
        return x


class ChannelAttention(layers.Layer):
    """Squeeze-and-Excitation Channel Attention module."""
    def __init__(self, channels, reduction=8, **kwargs):
        super().__init__(**kwargs)
        self.gap = layers.GlobalAveragePooling2D(keepdims=True)
        mid_channels = max(channels // reduction, 8)
        self.fc1 = layers.Conv2D(mid_channels, 1, activation="swish", padding="same")
        self.fc2 = layers.Conv2D(channels, 1, activation="sigmoid", padding="same")

    def call(self, x):
        attn = self.gap(x)
        attn = self.fc1(attn)
        attn = self.fc2(attn)
        return x * attn


class TextConditionedResBlock(layers.Layer):
    """
    Residual Block with FiLM (Feature-wise Linear Modulation) conditioning.
    Modulates feature maps using scale and shift derived from (time + text) conditioning.
    """
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.norm1 = layers.LayerNormalization(axis=-1, epsilon=1e-5)
        self.conv1 = layers.Conv2D(channels, 3, padding="same")
        self.act = layers.Activation("swish")

        # FiLM projection: outputs 2 * channels (scale and shift)
        self.cond_dense = layers.Dense(channels * 2, activation=None)

        self.norm2 = layers.LayerNormalization(axis=-1, epsilon=1e-5)
        self.conv2 = layers.Conv2D(channels, 3, padding="same")
        self.ca = ChannelAttention(channels)
        self.skip_conv = None

    def build(self, input_shape):
        x_shape = input_shape[0] if isinstance(input_shape, (list, tuple)) else input_shape
        in_channels = x_shape[-1]
        if in_channels != self.channels:
            self.skip_conv = layers.Conv2D(self.channels, 1, padding="same")
        super().build(input_shape)

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            x, cond_emb = inputs
        else:
            x, cond_emb = inputs, None

        res = self.norm1(x)
        res = self.act(res)
        res = self.conv1(res)

        # Apply FiLM modulation: h = h * (1 + scale) + shift
        if cond_emb is not None:
            mod = self.cond_dense(cond_emb)
            scale, shift = tf.split(mod, num_or_size_splits=2, axis=-1)
            scale = tf.cast(scale[:, None, None, :], res.dtype)
            shift = tf.cast(shift[:, None, None, :], res.dtype)
            res = res * (1.0 + scale) + shift

        res = self.norm2(res)
        res = self.act(res)
        res = self.conv2(res)
        res = self.ca(res)

        skip = self.skip_conv(x) if self.skip_conv is not None else x
        skip = tf.cast(skip, res.dtype)
        return skip + res


class TextContext(layers.Layer):
    """Projects token embeddings to the attention context; prepends an always-visible register token."""
    def __init__(self, dim=128, token_len=TOKEN_LEN, **kwargs):
        super().__init__(**kwargs)
        self.dim, self.token_len = dim, token_len
        self.proj = layers.Dense(dim)
        self.norm = layers.LayerNormalization(axis=-1, epsilon=1e-5)

    def build(self, input_shape):
        self.register = self.add_weight("register", shape=(1, 1, self.dim),
                                        initializer=tf.keras.initializers.RandomNormal(stddev=0.02))
        self.pos = self.add_weight("pos", shape=(1, self.token_len + 1, self.dim),
                                   initializer=tf.keras.initializers.RandomNormal(stddev=0.02))
        super().build(input_shape)

    def call(self, tok, mask):
        B = tf.shape(tok)[0]
        x = self.norm(self.proj(tok))
        x = x * tf.cast(mask[..., None], x.dtype)
        reg = tf.cast(tf.tile(self.register, [B, 1, 1]), x.dtype)
        x = tf.concat([reg, x], axis=1) + tf.cast(self.pos, x.dtype)
        m = tf.concat([tf.ones([B, 1], mask.dtype), mask], axis=1)
        return x, m


def _hw(x):
    """Static (H, W, C) when known, dynamic otherwise."""
    s, d = x.shape, tf.shape(x)
    return (s[1] or d[1]), (s[2] or d[2]), (s[3] or d[3])


class SelfAttention(layers.Layer):
    """Spatial self-attention with zero-init residual."""
    def __init__(self, channels, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.norm = layers.LayerNormalization(axis=-1, epsilon=1e-5)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=channels // num_heads)
        self.out = layers.Dense(channels, kernel_initializer="zeros")

    def call(self, x):
        B = tf.shape(x)[0]
        H, W, C = _hw(x)
        q = tf.reshape(self.norm(x), [B, H * W, C])
        a = self.attn(query=q, value=q, key=q)
        a = tf.reshape(self.out(a), [B, H, W, C])
        return x + tf.cast(a, x.dtype)


class CrossAttention(layers.Layer):
    """Spatial positions attend to all text tokens (masked), zero-init residual."""
    def __init__(self, channels, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.norm = layers.LayerNormalization(axis=-1, epsilon=1e-5)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=channels // num_heads)
        self.out = layers.Dense(channels, kernel_initializer="zeros")

    def call(self, x, ctx, ctx_mask):
        B = tf.shape(x)[0]
        H, W, C = _hw(x)
        q = tf.reshape(self.norm(x), [B, H * W, C])
        S = tf.shape(ctx)[1]
        am = tf.broadcast_to(ctx_mask[:, None, :] > 0.5, [B, H * W, S])
        a = self.attn(query=q, value=ctx, key=ctx, attention_mask=am)
        a = tf.reshape(self.out(a), [B, H, W, C])
        return x + tf.cast(a, x.dtype)


class TextToImageUNet(tf.keras.Model):
    """
    Text-conditioned diffusion U-Net v2.
    Inputs: [x_noisy (B,H,W,3), timesteps (B,), text (B, TOKEN_LEN, text_dim+1)]
    A rank-2 or missing text input is treated as the null (unconditional) prompt.
    """
    def __init__(self, base_filters=32, text_dim=384, time_dim=128, out_channels=3,
                 ctx_dim=128, token_len=TOKEN_LEN, attn32=False, **kwargs):
        super().__init__(**kwargs)
        f = base_filters
        c0, c1, c2, c3 = f, f * 2, f * 4, f * 8
        self.text_dim, self.token_len, self.attn32 = text_dim, token_len, attn32

        self.time_mlp = TimeEmbedding(dim=time_dim)
        self.text_mlp = TextProjection(in_dim=text_dim, out_dim=time_dim)
        self.text_ctx = TextContext(dim=ctx_dim, token_len=token_len)

        self.in_conv = layers.Conv2D(c0, 3, padding="same")

        # Encoder (128 -> 64 -> 32 -> 16)
        self.enc1_1 = TextConditionedResBlock(c0)
        self.down1 = layers.Conv2D(c1, 3, strides=2, padding="same")
        self.enc2_1 = TextConditionedResBlock(c1)
        self.enc2_2 = TextConditionedResBlock(c1)
        self.down2 = layers.Conv2D(c2, 3, strides=2, padding="same")
        self.enc3_1 = TextConditionedResBlock(c2)
        self.enc3_sa = SelfAttention(c2) if attn32 else None
        self.enc3_ca = CrossAttention(c2)
        self.enc3_2 = TextConditionedResBlock(c2)
        self.down3 = layers.Conv2D(c3, 3, strides=2, padding="same")

        # Bottleneck (16x16)
        self.mid_1 = TextConditionedResBlock(c3)
        self.mid_sa = SelfAttention(c3)
        self.mid_ca = CrossAttention(c3)
        self.mid_2 = TextConditionedResBlock(c3)

        # Decoder
        self.up3 = layers.Conv2DTranspose(c2, 3, strides=2, padding="same")
        self.dec3_fuse = layers.Conv2D(c2, 1, padding="same")
        self.dec3_1 = TextConditionedResBlock(c2)
        self.dec3_sa = SelfAttention(c2) if attn32 else None
        self.dec3_ca = CrossAttention(c2)
        self.dec3_2 = TextConditionedResBlock(c2)

        self.up2 = layers.Conv2DTranspose(c1, 3, strides=2, padding="same")
        self.dec2_fuse = layers.Conv2D(c1, 1, padding="same")
        self.dec2_1 = TextConditionedResBlock(c1)
        self.dec2_2 = TextConditionedResBlock(c1)

        self.up1 = layers.Conv2DTranspose(c0, 3, strides=2, padding="same")
        self.dec1_fuse = layers.Conv2D(c0, 1, padding="same")
        self.dec1_1 = TextConditionedResBlock(c0)
        self.dec1_2 = TextConditionedResBlock(c0)

        self.out_norm = layers.LayerNormalization(axis=-1, epsilon=1e-5)
        self.out_act = layers.Activation("swish")
        self.out_conv = layers.Conv2D(out_channels, 3, padding="same", dtype="float32")

    def _split_text(self, text, B):
        if text is None or len(text.shape) != 3:
            tok = tf.zeros([B, self.token_len, self.text_dim], tf.float32)
            mask = tf.zeros([B, self.token_len], tf.float32)
        else:
            text = tf.cast(text, tf.float32)
            tok, mask = text[..., :-1], tf.cast(text[..., -1] > 0.5, tf.float32)
        return tok, mask

    def call(self, inputs, training=None):
        x = inputs[0]
        t = inputs[1] if len(inputs) > 1 else tf.zeros([tf.shape(x)[0]], tf.int32)
        text = inputs[2] if len(inputs) > 2 else None
        B = tf.shape(x)[0]
        tok, mask = self._split_text(text, B)

        # global conditioning (time + masked-mean text) for FiLM
        denom = tf.maximum(tf.reduce_sum(mask, axis=1, keepdims=True), 1.0)
        pooled = tf.math.l2_normalize(tf.reduce_sum(tok * mask[..., None], axis=1) / denom, axis=-1)
        cond = self.time_mlp(t) + self.text_mlp(pooled)
        # token context for cross-attention
        ctx, cmask = self.text_ctx(tok, mask)

        x0 = self.in_conv(x)
        e1 = self.enc1_1([x0, cond])
        e2 = self.enc2_2([self.enc2_1([self.down1(e1), cond]), cond])
        h = self.enc3_1([self.down2(e2), cond])
        if self.enc3_sa is not None:
            h = self.enc3_sa(h)
        h = self.enc3_ca(h, ctx, cmask)
        e3 = self.enc3_2([h, cond])

        b = self.mid_1([self.down3(e3), cond])
        b = self.mid_sa(b)
        b = self.mid_ca(b, ctx, cmask)
        b = self.mid_2([b, cond])

        u3 = self.dec3_fuse(tf.concat([tf.cast(self.up3(b), e3.dtype), e3], axis=-1))
        h = self.dec3_1([u3, cond])
        if self.dec3_sa is not None:
            h = self.dec3_sa(h)
        h = self.dec3_ca(h, ctx, cmask)
        d3 = self.dec3_2([h, cond])

        u2 = self.dec2_fuse(tf.concat([tf.cast(self.up2(d3), e2.dtype), e2], axis=-1))
        d2 = self.dec2_2([self.dec2_1([u2, cond]), cond])

        u1 = self.dec1_fuse(tf.concat([tf.cast(self.up1(d2), e1.dtype), e1], axis=-1))
        d1 = self.dec1_2([self.dec1_1([u1, cond]), cond])

        out = self.out_conv(self.out_act(self.out_norm(d1)))
        return tf.cast(out, tf.float32)


# Alias for backward compatibility
DiffusionUNet = TextToImageUNet
