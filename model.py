import os
import glob
from PIL import Image
from tqdm.auto import tqdm
import jax
import itertools
import numpy as np
import wandb
import jax.numpy as jnp
import equinox as eqx
import kagglehub
import optax
import matplotlib.pyplot as plt
        
def beta_schedule(beta_start, beta_end, timesteps):
    betas = jnp.linspace(beta_start, beta_end, timesteps)
    alphas = 1.0 - betas
    alpha_bars = jnp.cumprod(alphas)
    return betas, alphas, alpha_bars

def timestep_embedding(t, embed_dim=128):
    half_dim = embed_dim // 2
    exp = jnp.linspace(0, jnp.log(10000.0), half_dim)
    freqs = jnp.exp(-exp)
    args = t[:, None] * freqs[None, :]
    emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
    return emb

class SinusoidalTimeStepEmbedding(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, key, timestep_embed_dim=128, embed_dim=32):
        self.linear = eqx.filter_vmap(
            eqx.nn.Linear(in_features=timestep_embed_dim, out_features=embed_dim, key=key)
        )

    def __call__(self, t):
        emb = timestep_embedding(t, embed_dim=128)
        return jax.nn.relu(self.linear(emb))

class ConvBlock(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d

    def __init__(self, key, in_channel_dim, conv_embed_dim):
        k1, k2 = jax.random.split(key)
        self.conv1 = eqx.nn.Conv2d(in_channels=in_channel_dim, out_channels=conv_embed_dim, kernel_size=3, key=k1, padding='same')
        self.conv2 = eqx.nn.Conv2d(in_channels=conv_embed_dim, out_channels=conv_embed_dim, kernel_size=3, key=k2, padding='same')

    def __call__(self, x):
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))
        return x

class UNet(eqx.Module):
    down1: ConvBlock
    down2: ConvBlock
    down3: ConvBlock
    bottleneck: ConvBlock
    up1: ConvBlock
    up2: ConvBlock
    up3: ConvBlock
    final_conv: eqx.nn.Conv2d
    t_proj: eqx.nn.Linear
    t_embed: SinusoidalTimeStepEmbedding

    def __init__(self, key):
        k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = jax.random.split(key, 10)
        self.down1 = ConvBlock(k1, 3, 64)
        self.down2 = ConvBlock(k2, 64, 128)
        self.down3 = ConvBlock(k3, 128, 256)
        self.bottleneck = ConvBlock(k4, 256, 512)
        self.up1 = ConvBlock(k5, 512 + 256, 256)
        self.up2 = ConvBlock(k6, 256 + 128, 128)
        self.up3 = ConvBlock(k7, 128 + 64, 64)
        self.final_conv = eqx.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, key=k8)
    
        self.t_embed = SinusoidalTimeStepEmbedding(k9, embed_dim=64)
        self.t_proj = eqx.nn.Linear(64, 256, key=k10)

        self.down1 = eqx.filter_vmap(self.down1)
        self.down2 = eqx.filter_vmap(self.down2)
        self.down3 = eqx.filter_vmap(self.down3)
        self.bottleneck = eqx.filter_vmap(self.bottleneck)
        self.up1 = eqx.filter_vmap(self.up1)
        self.up2 = eqx.filter_vmap(self.up2)
        self.up3 = eqx.filter_vmap(self.up3)
        self.final_conv = eqx.filter_vmap(self.final_conv)
        self.t_proj = eqx.filter_vmap(self.t_proj)

    def __call__(self, x, t):
        t_emb = self.t_embed(t)

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        t_emb_proj = self.t_proj(t_emb)
        h_t = x3 + t_emb_proj[:, :, None, None]
        b = self.bottleneck(h_t)

        x = self.up1(jnp.concatenate([b, x3], axis=1))
        x = self.up2(jnp.concatenate([x, x2], axis=1))
        x = self.up3(jnp.concatenate([x, x1], axis=1))
        return self.final_conv(x)
