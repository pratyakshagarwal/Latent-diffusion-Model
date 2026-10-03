
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.loss import PerceptualLoss, elbo
from src.variational_autoencoder import VAE


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

IN_CHANNELS = 3
OUT_CHANNELS = 64
LATENT_DIM = 128
INPUT_SIZE = (32, 32)

BATCH_SIZE = 128
LEARNING_RATE = 1e-4
EPOCHS = 50
KL_WARMUP_EPOCHS = 10

DATA_DIR = "./data"
CHECKPOINT_PATH = "models/vae_cnn_cifar10latent128_epoch50.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------
# Dataset and DataLoader
# ------------------------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),])

dataset = datasets.CIFAR10(root=DATA_DIR,  train=True, download=True, transform=transform,)
dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# ------------------------------------------------------------------
# Model, Loss, and Optimizer
# ------------------------------------------------------------------

MODEL = VAE(
    in_channels=IN_CHANNELS,
    out_channels=OUT_CHANNELS,
    latent_dim=LATENT_DIM,
    input_size=INPUT_SIZE,
).to(DEVICE)

PLOSS = PerceptualLoss()

OPTIMIZER = torch.optim.Adam(
    MODEL.parameters(),
    lr=LEARNING_RATE,
)

# Cosine decay over all training steps, not just epochs.
T_MAX = EPOCHS * len(dl)

LR_SCHEDULER = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=OPTIMIZER,
    T_max=T_MAX,
)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_vae(
    model,
    dl,
    epochs,
    optimizer,
    lr_scheduler,
    device,
    ploss: callable,
):
    """Train the VAE with ELBO loss and KL warmup."""

    print(f"Using device: {device}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        # Gradually increase the KL contribution during warmup.
        kl_weight = min(1.0, epoch / KL_WARMUP_EPOCHS)

        for xb, _ in dl:
            xb = xb.to(device)
            optimizer.zero_grad()
            x_hat, mu, log_var = model(xb)

            loss = elbo(xb, x_hat, mu, log_var,
                loss_fn="mse", perceptual_fn=ploss,
                verbose=False, kl_weight=kl_weight,
            )

            loss.backward()

            # Prevent excessively large gradients.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dl)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"KL Weight: {kl_weight:.2f}"
        )


# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------

if __name__ == "__main__":
    train_vae(
        MODEL,
        dl,
        EPOCHS,
        OPTIMIZER,
        LR_SCHEDULER,
        DEVICE,
        PLOSS,
    )

    # Ensure the checkpoint directory exists.
    checkpoint = Path(CHECKPOINT_PATH)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    torch.save(MODEL.state_dict(), checkpoint)

    print(f"Model saved to: {checkpoint}")