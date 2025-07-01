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
from model import UNet, beta_schedule


def forward_step(x_0, t, alpha_bars, noise):
    alpha_bars_sqrt = jnp.sqrt(alpha_bars[t])[:, None, None, None]
    x_t = alpha_bars_sqrt[t] * x_0 + (jnp.sqrt(1 - alpha_bars[t])[:, None, None, None]) * noise
    return x_t

def loss_fn(model, x0, t, alpha_bars, key):
    noise = jax.random.normal(key, x0.shape)
    x_t = forward_step(x0, t, alpha_bars, noise)
    pred_noise = model(x_t, t)
    return jnp.mean((noise - pred_noise) ** 2)

def ema_update(ema_params, current_params, decay):
    return jax.tree_util.tree_map(
        lambda e, p: decay * e + (1. - decay) * p,
        ema_params, current_params
    )

@eqx.filter_value_and_grad
def compute_loss(model, x0, t, alpha_bars, key):
    return loss_fn(model, x0, t, alpha_bars, key)

@eqx.filter_jit
def train_step(model, x, t, key, alpha_bars, optimizer, opt_state):
    loss, grads = compute_loss(model, x, t, alpha_bars, key)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

def train(model, dataloader, optimizer, alphas, alpha_bars, betas, key, steps=50000):
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    ema_model = model
    decay = 0.999

    for step, x in tqdm(enumerate(dataloader), total=steps):
        if step >= steps:
            break
        key, sample_key, time_key = jax.random.split(key, 3)
        t = jax.random.randint(time_key, (x.shape[0],), 0, len(alpha_bars))

        model, opt_state, loss = train_step(model, x, t, key, alpha_bars, optimizer, opt_state)

        ema_model = eqx.combine(ema_model, ema_update(
            eqx.filter(ema_model, eqx.is_array),
            eqx.filter(model, eqx.is_array),
            decay
        ))

        wandb.log({"loss": loss.item()}, step=int(step))

        if step % 100 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")

        if step % 500 == 0:
            samples = sample(
                ema_model, (16, 3, 64, 64), betas, alphas, alpha_bars, sample_key
            )
            log_images(samples, step)

    return model

def p_sample(model, x_t, timestep, alphas, alpha_bars, betas, key):
    if timestep > 0:
        z = jax.random.normal(key, x_t.shape)
    else:
        z = jnp.array(0.0)

    pred_noise = model(x_t, jnp.array([timestep]))
    alpha, alpha_bar, beta = alphas[timestep], alpha_bars[timestep], betas[timestep]

    coef1 = 1 / jnp.sqrt(alpha)
    coef2 = (1 - alpha) / jnp.sqrt(1 - alpha_bar)

    mean = coef1 * (x_t - coef2 * pred_noise)
    return mean + jnp.sqrt(beta) * z

def sample(model, shape, betas, alphas, alpha_hats, key):
    x = jax.random.normal(key, shape)
    for t in reversed(range(len(betas))):
        key, subkey = jax.random.split(key)
        x = p_sample(model, x, t, alphas, alpha_hats, betas, subkey)
    return x

def log_images(images, step):
    images = (images + 1) * 0.5
    grid = np.clip(np.transpose(images[:16], (0, 2, 3, 1)), 0, 1)
    wandb.log({"samples": [wandb.Image(np.array(img)) for img in grid]}, step=step)

imgs_path = os.path.join(os.getcwd(), "train_images")

def preprocess_jax(img, image_size=32):
    img = Image.fromarray(np.array(img)).resize((image_size, image_size), resample=Image.BICUBIC)
    img = jnp.array(img).astype(jnp.float32) / 255.0
    img = img * 2.0 - 1.0
    return jnp.transpose(img, (2, 0, 1))

def pil_transform(img, image_size=64):
    img = img.resize((image_size, image_size), resample=Image.BICUBIC)
    img = jnp.array(img).astype(jnp.float32) / 255.0
    img = img * 2.0 - 1.0
    img = jnp.transpose(img, (2, 0, 1))
    return img

class Cifar10Dataset:
    def __init__(self, imgs_path, transform=None, sample_size=2500, seed=42):
        self.imgs_path = imgs_path
        self.transform = transform
        self.sample_size = sample_size

        self.imgs_path_list = glob.glob(os.path.join(self.imgs_path, "*.png"))
        # Shuffle and sample
        rng = np.random.default_rng(seed)
        self.imgs_path_list = rng.permutation(self.imgs_path_list)[:sample_size]
    
    def __len__(self):
        return len(self.imgs_path_list)
    
    def __getitem__(self, idx):
        img_path = self.imgs_path_list[idx]
        img = Image.open(img_path).convert("RGB")
        img = jnp.array(img)

        if self.transform is not None:
            return self.transform(img)
        else:
            return img

class CelebADataset:
    def __init__(self, imgs_path, transform=None, sample_size=10000, seed=42):
        self.imgs_path = imgs_path
        self.transform = transform
        self.sample_size = sample_size

        self.imgs_path_list = glob.glob(os.path.join(self.imgs_path, "*.jpg"))
        # Shuffle and sample
        rng = np.random.default_rng(seed)
        self.imgs_path_list = rng.permutation(self.imgs_path_list)[:sample_size]

    def __len__(self):
        return len(self.imgs_path_list)

    def __getitem__(self, idx):
        img_path = self.imgs_path_list[idx]
        img = Image.open(img_path).convert("RGB")
        img = jnp.array(img)

        if self.transform is not None:
            return self.transform(img)
        else:
            return img

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.idx = 0
        self.len = len(self.dataset)
        self.fetch_idxs = jnp.arange(self.len)

        if self.shuffle:
            key = jax.random.PRNGKey(0)
            self.fetch_idxs = jax.random.permutation(key, self.fetch_idxs)

        if self.drop_last:
            # This is the crucial part
            remainder = len(self.fetch_idxs) % self.batch_size
            if remainder != 0:
                self.fetch_idxs = self.fetch_idxs[:-remainder]

        self.num_batches = len(self.fetch_idxs) // self.batch_size
        self.fetch_idxs = jnp.split(self.fetch_idxs, self.num_batches)

    def __iter__(self):
        self.idx = 0
        return self

    def __len__(self):
        return self.num_batches

    def __next__(self):
        if self.idx >= len(self.fetch_idxs):
            raise StopIteration
        else:
            batch_indices = self.fetch_idxs[self.idx]
            batch_imgs = jnp.stack([self.dataset[i] for i in batch_indices])
            self.idx += 1
            return batch_imgs

if __name__ == "__main__":
    print(f"Active JAX devices: {jax.devices()}")

    model = UNet(jax.random.PRNGKey(0))

    dataset = Cifar10Dataset(imgs_path, transform=preprocess_jax)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    dataloader = itertools.cycle(dataloader)

    betas, alphas, alpha_bars = beta_schedule(0.0001, 0.02, 1000)

    optimizer = optax.adamw(learning_rate=2e-5)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    dummy_x = jnp.zeros((16, 3, 32, 32), dtype=jnp.float32)
    dummy_t = jnp.zeros((16,), dtype=jnp.int32)
    dummy_key = jax.random.PRNGKey(42)

    _ = train_step(model, dummy_x, dummy_t, dummy_key, alpha_bars, optimizer, opt_state)

    _ = sample(model, (16, 3, 64, 64), betas, alphas, alpha_bars, dummy_key)

    wandb.init(project="nano-ddpm", name="cifar10-run-TPU-v6e1", config={"image_size": 64, "T": 1000})

    train(model, dataloader, optimizer, alphas, alpha_bars, betas, steps=10000, key=jax.random.PRNGKey(0))

    wandb.finish()