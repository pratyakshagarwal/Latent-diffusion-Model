import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from src.scheduler import NoiseScheduler
from src.unet import Unet
from src.variational_autoencoder import VAE


# -- Paths --
VAE_PATH = "models/vae_cnn_cifar10latent128_epoch50.pth"
MODEL_PATH = "models/latent_ddpm_dim64.pth"
LATENT_PATH = "data/cifar10/latent128_train.pt"

# -- VAE Params --
VAE_IN_CHANNELS = 3
VAE_OUT_CHANNELS = 64
LATENT_DIM = 128
INPUT_SIZE = (32, 32)

# -- U-Net Params --
UNET_IN_CHANNELS = 8
UNET_OUT_CHANNELS = 256
KERNEL_SIZE = 3
TIME_DIM = 128
NUM_GROUPS = 32

# -- Sampling Params --
NUM_TIMESTEPS = 1000
NULL_CLASS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-label", type=int, default=0, choices=range(10))
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--output", type=str, default="plots/sample.png")
    return parser.parse_args()


def load_models():
    vae = VAE(
        in_channels=VAE_IN_CHANNELS,
        out_channels=VAE_OUT_CHANNELS,
        latent_dim=LATENT_DIM,
        input_size=INPUT_SIZE
    ).to(DEVICE)

    vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
    vae.eval()

    model = Unet(
        in_channels=UNET_IN_CHANNELS,
        out_channel=UNET_OUT_CHANNELS,
        kernel_size=KERNEL_SIZE,
        time_dim=TIME_DIM,
        num_groups=NUM_GROUPS
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    return vae, model


def load_latent_stats():
    data = torch.load(LATENT_PATH, map_location="cpu")
    return data["mean"].to(DEVICE), data["std"].to(DEVICE)


@torch.no_grad()
def sample(model, vae, scheduler, mean, std, n_samples=8,
           class_label=0, guidance_scale=7.5):

    betas = scheduler.betas.to(DEVICE)
    alphas = scheduler.alphas.to(DEVICE)
    alpha_bars = scheduler.alpha_bars.to(DEVICE)

    class_labels = torch.full(
        (n_samples,), class_label, dtype=torch.long, device=DEVICE
    )
    null_labels = torch.full(
        (n_samples,), NULL_CLASS, dtype=torch.long, device=DEVICE
    )

    # Start from Gaussian noise in latent space.
    xt = torch.randn(n_samples, UNET_IN_CHANNELS, 8, 8, device=DEVICE)

    for t in range(NUM_TIMESTEPS - 1, -1, -1):
        t_batch = torch.full(
            (n_samples,), t, dtype=torch.long, device=DEVICE
        )

        pred_noise_uncond = model(xt, t_batch, null_labels)
        pred_noise_cond = model(xt, t_batch, class_labels)

        # Classifier-free guidance.
        pred_noise = (
            pred_noise_uncond
            + guidance_scale * (pred_noise_cond - pred_noise_uncond)
        )

        z = torch.randn_like(xt) if t > 0 else torch.zeros_like(xt)

        sigma_t = torch.sqrt(betas[t])
        denom = torch.sqrt(alphas[t]) + 1e-8

        xt = (
            (1 / denom)
            * (
                xt
                - (betas[t] / torch.sqrt(1 - alpha_bars[t] + 1e-8))
                * pred_noise
            )
            + sigma_t * z
        )

    # Undo latent normalization before decoding.
    xt = xt * std + mean

    return vae.decoder(xt)


def plot_samples(samples, class_label, guidance_scale, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    n_samples = samples.shape[0]
    fig, axes = plt.subplots(1, n_samples, figsize=(2 * n_samples, 2))

    if n_samples == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        img = samples[i].cpu().numpy().transpose(1, 2, 0).clip(0, 1)
        ax.imshow(img)
        ax.axis("off")

    fig.suptitle(
        f"{CLASS_NAMES[class_label]} | CFG: {guidance_scale}"
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    args = parse_args()

    print(f"Using: {DEVICE}")
    print(
        f"Class: {args.class_label} ({CLASS_NAMES[args.class_label]}) | "
        f"CFG: {args.guidance_scale}"
    )

    vae, model = load_models()
    mean, std = load_latent_stats()
    scheduler = NoiseScheduler()

    samples = sample(
        model=model,
        vae=vae,
        scheduler=scheduler,
        mean=mean,
        std=std,
        n_samples=args.num_samples,
        class_label=args.class_label,
        guidance_scale=args.guidance_scale
    )

    plot_samples(
        samples,
        args.class_label,
        args.guidance_scale,
        args.output
    )