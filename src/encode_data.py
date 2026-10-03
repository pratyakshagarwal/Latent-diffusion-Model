
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.variational_autoencoder import VAE


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

MODEL_PATH = "../models/vae_cnn_cifar10latent128_epoch50.pth"
LATENT_PATH = "../data/cifar10/latent128_train.pt"

IN_CHANNELS = 3
OUT_CHANNELS = 64
LATENT_DIM = 128
INPUT_SIZE = (32, 32)

BATCH_SIZE = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------
# Dataset and DataLoader
# ------------------------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform,)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)  # Preserve dataset order during encoding.


# ------------------------------------------------------------------
# Load Trained VAE
# ------------------------------------------------------------------

vae = VAE(
    in_channels=IN_CHANNELS,
    out_channels=OUT_CHANNELS,
    latent_dim=LATENT_DIM,
    input_size=INPUT_SIZE,
).to(DEVICE)

vae.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

vae.eval()


# ------------------------------------------------------------------
# Encode Dataset into Latent Representations
# ------------------------------------------------------------------

def encode_cifar10(model, dataloader, path):
    """Encode CIFAR-10 images and save normalized latents and labels."""

    latents, labels = [], []

    model.eval()

    with torch.no_grad():
        for images, y in dataloader:
            images = images.to(DEVICE)

            # Use the encoder's mean as the latent representation.
            mu, _ = model.encoder(images)

            latents.append(mu.cpu())
            labels.append(y)

    latents = torch.cat(latents, dim=0)
    labels = torch.cat(labels, dim=0)

    # Standardize latents for downstream diffusion training.
    mean = latents.mean()
    std = latents.std()

    latents = (latents - mean) / (std + 1e-8)

    # Ensure the destination directory exists.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "latents": latents,
            "labels": labels,
            "mean": mean,
            "std": std,
        },
        path,
    )

    print(f"Latents shape: {latents.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Latent mean: {mean.item():.4f}")
    print(f"Latent std: {std.item():.4f}")
    print(f"Saved latent dataset to: {path}")


# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------

if __name__ == "__main__":
    encode_cifar10(
        vae,
        dataloader,
        path=LATENT_PATH,
    )