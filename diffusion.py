"""
Denoising Diffusion Probabilistic Models (DDPM) & Fast DDIM Engine.
Supports:
- Forward Noising q(x_t | x_0)
- Text-to-Image Generation with Classifier-Free Guidance (CFG)
- Image Restoration / Reverse Denoising
- Cosine Variance Schedule
"""

import math
import numpy as np
import tensorflow as tf


def cosine_beta_schedule(timesteps=1000, s=0.008):
    """
    Cosine variance schedule (Nichol & Dhariwal, 2021).
    Prevents abrupt noise changes near t=0 and t=T.
    """
    steps = timesteps + 1
    t = np.linspace(0, timesteps, steps, dtype=np.float32) / timesteps
    alphas_cumprod = np.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0.0001, 0.999)


class GaussianDiffusion:
    """Core diffusion engine for training and text-to-image sampling."""
    def __init__(self, timesteps=1000, schedule="cosine"):
        self.timesteps = timesteps

        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = np.linspace(1e-4, 0.02, timesteps, dtype=np.float32)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.betas = tf.constant(betas, dtype=tf.float32)
        self.alphas = tf.constant(alphas, dtype=tf.float32)
        self.alphas_cumprod = tf.constant(alphas_cumprod, dtype=tf.float32)
        self.alphas_cumprod_prev = tf.constant(alphas_cumprod_prev, dtype=tf.float32)

        self.sqrt_alphas_cumprod = tf.constant(np.sqrt(alphas_cumprod), dtype=tf.float32)
        self.sqrt_one_minus_alphas_cumprod = tf.constant(np.sqrt(1.0 - alphas_cumprod), dtype=tf.float32)

    def _extract(self, a, t, x_shape):
        """Extract coefficients at specified timesteps t."""
        batch_size = tf.shape(t)[0]
        out = tf.gather(a, t)
        return tf.reshape(out, [batch_size] + [1] * (len(x_shape) - 1))

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise."""
        if noise is None:
            noise = tf.random.normal(tf.shape(x_start))

        sqrt_alphas_t = self._extract(self.sqrt_alphas_cumprod, t, tf.shape(x_start))
        sqrt_one_minus_alphas_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, tf.shape(x_start))

        return sqrt_alphas_t * x_start + sqrt_one_minus_alphas_t * noise

    def sample_text_to_image(
        self,
        model,
        text_emb,
        shape,
        guidance_scale=7.5,
        ddim_steps=50,
        eta=0.0,
    ):
        """
        Generates images from text prompt embeddings using Classifier-Free Guidance (CFG).
        
        Args:
            model: Trained TextToImageUNet.
            text_emb: Text embedding tensor of shape (B, text_dim).
            shape: Target image shape [B, H, W, 3].
            guidance_scale: Strength of prompt conditioning (1.0 = standard, 5.0-8.0 = strong).
            ddim_steps: Number of reverse diffusion steps (20-50 recommended).
            eta: 0.0 for deterministic sampling.
        """
        batch_size = shape[0]
        step_stride = max(self.timesteps // ddim_steps, 1)
        time_steps = list(range(0, self.timesteps, step_stride))
        time_steps = np.flip(time_steps)

        # Start from pure Gaussian noise
        img = tf.random.normal(shape)
        null_emb = tf.zeros_like(text_emb)

        for i, step in enumerate(time_steps):
            t = tf.fill([batch_size], step)
            alpha_bar_t = self._extract(self.alphas_cumprod, t, tf.shape(img))

            if i < len(time_steps) - 1:
                next_step = time_steps[i + 1]
                t_next = tf.fill([batch_size], next_step)
                alpha_bar_next = self._extract(self.alphas_cumprod, t_next, tf.shape(img))
            else:
                alpha_bar_next = tf.ones_like(alpha_bar_t)

            # Predict noise with text prompt (conditioned)
            if guidance_scale > 1.0:
                # one batched forward pass for conditional + unconditional
                both = model(
                    [tf.concat([img, img], 0), tf.concat([t, t], 0), tf.concat([text_emb, null_emb], 0)],
                    training=False,
                )
                pred_noise_cond, pred_noise_uncond = tf.split(both, 2, axis=0)
                pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)
            else:
                pred_noise = model([img, t, text_emb], training=False)

            # Tweedie's formula to estimate x0
            pred_x0 = (img - tf.sqrt(tf.maximum(1.0 - alpha_bar_t, 1e-6)) * pred_noise) / tf.sqrt(
                tf.maximum(alpha_bar_t, 1e-6)
            )
            pred_x0 = tf.clip_by_value(pred_x0, -1.0, 1.0)

            # Direction toward x_{t-1}
            sigma = eta * tf.sqrt(
                (1.0 - alpha_bar_next) / (1.0 - alpha_bar_t) * (1.0 - alpha_bar_t / alpha_bar_next)
            )
            dir_xt = tf.sqrt(tf.maximum(1.0 - alpha_bar_next - sigma ** 2, 0.0)) * pred_noise

            noise = tf.random.normal(tf.shape(img)) if eta > 0.0 else 0.0
            img = tf.sqrt(alpha_bar_next) * pred_x0 + dir_xt + sigma * noise

        return tf.clip_by_value(img, -1.0, 1.0)

    def restore_image(self, model, noisy_img, start_step=250, ddim_steps=15):
        """Restores / denoises an image using reverse DDIM."""
        batch_size = tf.shape(noisy_img)[0]
        step_stride = max(start_step // ddim_steps, 1)
        time_steps = list(range(0, start_step + 1, step_stride))
        time_steps = np.flip(time_steps)

        img = tf.identity(noisy_img)
        dummy_text = tf.zeros([batch_size, 384], dtype=tf.float32)

        for i, step in enumerate(time_steps):
            t = tf.fill([batch_size], step)
            pred_noise = model([img, t, dummy_text], training=False)
            alpha_bar_t = self._extract(self.alphas_cumprod, t, tf.shape(img))

            if i < len(time_steps) - 1:
                next_step = time_steps[i + 1]
                t_next = tf.fill([batch_size], next_step)
                alpha_bar_next = self._extract(self.alphas_cumprod, t_next, tf.shape(img))
            else:
                alpha_bar_next = tf.ones_like(alpha_bar_t)

            pred_x0 = (img - tf.sqrt(tf.maximum(1.0 - alpha_bar_t, 1e-6)) * pred_noise) / tf.sqrt(
                tf.maximum(alpha_bar_t, 1e-6)
            )
            pred_x0 = tf.clip_by_value(pred_x0, -1.0, 1.0)

            dir_xt = tf.sqrt(tf.maximum(1.0 - alpha_bar_next, 0.0)) * pred_noise
            img = tf.sqrt(alpha_bar_next) * pred_x0 + dir_xt

        return tf.clip_by_value(img, -1.0, 1.0)
