
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from src.scheduler import NoiseScheduler
from src.unet import Unet


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENT_PATH = "../data/cifar10/latent128_train.pt"
CHECKPOINT_DIR = "../models"
MODEL_NAME = "latent_ddpm_dim64"

# VAE configuration (used when creating the latent dataset).
VAE_MODEL_PATH = "../models/vae_cnn_cifar10latent128_epoch50.pth"
VAE_IN_CHANNELS = 3
VAE_OUT_CHANNELS = 64
VAE_LATENT_DIM = 128
VAE_INPUT_SIZE = (32, 32)

# U-Net configuration.
UNET_IN_CHANNELS = 8
UNET_OUT_CHANNELS = 256
KERNEL_SIZE = 3
TIME_DIM = 128
NUM_GROUPS = 32

# Training configuration.
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 100
NUM_TIMESTEPS = 1000

# Classifier-free guidance training configuration.
NULL_CLASS = 10
UNCOND_PROB = 0.1

# Train/validation split.
TRAIN_RATIO = 0.9
RANDOM_SEED = 42


# ------------------------------------------------------------------
# Latent Dataset
# ------------------------------------------------------------------

class CIFAR10LatentDataset(Dataset):
    """Dataset containing precomputed CIFAR-10 latents and labels."""

    def __init__(self, data_path: str): self.data = torch.load(data_path, map_location="cpu")

    def __getitem__(self, index: int):
        return (
            self.data["latents"][index],
            self.data["labels"][index],
        )

    def __len__(self): return len(self.data["latents"])


# ------------------------------------------------------------------
# Training and Validation
# ------------------------------------------------------------------

def train(
    model,
    noise_scheduler,
    train_dl,
    val_dl,
    optimizer,
    epochs,
    device,
    lr_scheduler,
    model_name: str = "ddpm",
):
    """Train a latent DDPM with classifier-free label dropout."""

    print(f"Using device: {device}")

    best_val_loss = float("inf")
    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"{model_name}.pth"

    for epoch in range(1, epochs + 1):

        # ----------------------------------------------------------
        # Training
        # ----------------------------------------------------------

        model.train()
        train_loss = 0.0

        for x0, yb in train_dl:
            x0 = x0.to(device)
            yb = yb.to(device)

            # Sample a random diffusion timestep for each latent.
            t = torch.randint(
                0,
                NUM_TIMESTEPS,
                (x0.shape[0],),
                device=device,
            )

            # Forward diffusion: add noise to the clean latent.
            true_noise = torch.randn_like(x0)
            noisy_x0 = noise_scheduler.add_noise(x0, t, true_noise)

            # Randomly replace labels with the null class.
            # This enables classifier-free guidance during sampling.
            mask = torch.rand(x0.shape[0], device=device) < UNCOND_PROB
            yb = yb.clone()
            yb[mask] = NULL_CLASS

            # Predict the noise added to the latent.
            optimizer.zero_grad()

            pred_noise = model(noisy_x0, t, yb)
            loss = F.mse_loss(pred_noise, true_noise)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            lr_scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dl)

        # ----------------------------------------------------------
        # Validation
        # ----------------------------------------------------------

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x0, yb in val_dl:
                x0 = x0.to(device)
                yb = yb.to(device)

                t = torch.randint(
                    0,
                    NUM_TIMESTEPS,
                    (x0.shape[0],),
                    device=device,
                )

                true_noise = torch.randn_like(x0)
                noisy_x0 = noise_scheduler.add_noise(x0, t, true_noise)

                pred_noise = model(noisy_x0, t, yb)
                loss = F.mse_loss(pred_noise, true_noise)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dl)

        # ----------------------------------------------------------
        # Checkpointing
        # ----------------------------------------------------------

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            torch.save(model.state_dict(), checkpoint_path)

            print(
                f"Saved best model at epoch {epoch} "
                f"(val loss: {avg_val_loss:.4f})"
            )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train: {avg_train_loss:.4f} | "
            f"Val: {avg_val_loss:.4f}"
        )


# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------

if __name__ == "__main__":

    # Load the precomputed latent representations.
    dataset = CIFAR10LatentDataset(LATENT_PATH)

    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size

    # Reproducible train/validation split.
    generator = torch.Generator().manual_seed(RANDOM_SEED)

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Initialize the latent-space U-Net.
    unet = Unet(
        in_channels=UNET_IN_CHANNELS,
        out_channel=UNET_OUT_CHANNELS,
        kernel_size=KERNEL_SIZE,
        time_dim=TIME_DIM,
        num_groups=NUM_GROUPS,
    ).to(DEVICE)

    num_params = sum(
        p.numel() for p in unet.parameters() if p.requires_grad
    )

    print(f"Trainable parameters: {num_params:,}")

    optimizer = torch.optim.Adam(
        unet.parameters(),
        lr=LEARNING_RATE,
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=EPOCHS * len(train_dl),
    )

    noise_scheduler = NoiseScheduler()

    train(
        model=unet,
        noise_scheduler=noise_scheduler,
        train_dl=train_dl,
        val_dl=val_dl,
        optimizer=optimizer,
        epochs=EPOCHS,
        device=DEVICE,
        lr_scheduler=lr_scheduler,
        model_name=MODEL_NAME,
    )