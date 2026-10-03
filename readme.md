# Latent Diffusion Model on CIFAR-10

A class-conditional Latent Diffusion Model implemented from scratch in PyTorch and trained on CIFAR-10.

The project follows the core idea of Stable Diffusion: instead of running the diffusion process directly on high-dimensional RGB images, images are first compressed into a lower-dimensional latent space using a VAE. The diffusion model then learns to denoise these latents.

## What This Is

A minimal end-to-end LDM pipeline:

* VAE compresses CIFAR-10 images from `(3, 32, 32)` into spatial latents of `(8, 8, 8)`.
* A DDPM U-Net learns the diffusion process entirely in latent space.
* Class embeddings make the diffusion model class-conditional.
* Classifier-Free Guidance (CFG) enables controllable class-conditioned generation.
* Generated latents are denormalized and passed through the VAE decoder to produce images.

The implementation is built to understand the individual components rather than relying on an existing diffusion framework.

## Pipeline

```text
CIFAR-10 Image
      │
      ▼
     VAE
      │
      ▼
 (8, 8, 8) Latent
      │
      ▼
 Add Noise ──────────────┐
      │                  │
      ▼                  │
 Latent U-Net            │
      │                  │
      ▼                  │
 Predicted Noise         │
      │                  │
      └── Reverse DDPM ◄─┘
              │
              ▼
      Denoised Latent
              │
       Unnormalize
              │
              ▼
         VAE Decoder
              │
              ▼
       Generated Image
```

## Architecture

### VAE

The VAE learns a compact spatial representation of CIFAR-10 images.

* Encoder: convolutional layers with ResBlocks and self-attention at the bottleneck
* Latent representation: `(8, 8, 8)`
* Decoder: ConvTranspose layers mirroring the encoder
* Loss: reconstruction MSE + VGG16 perceptual loss + KL divergence
* KL regularization uses free bits / KL warmup during training

The encoder's `mu` is used as the deterministic latent representation for the downstream diffusion model.

### Latent DDPM U-Net

The diffusion model operates on the VAE latent representation rather than directly on pixels.

* Input: `(8, 8, 8)` latent
* Time embedding
* Class embedding for CIFAR-10 classes
* ResBlocks with GroupNorm
* Self-attention at the bottleneck
* Noise prediction objective using MSE
* Classifier-Free Guidance during sampling

During training, a null class is randomly substituted for the actual class label. This allows the same model to learn both conditional and unconditional noise predictions.

### Classifier-Free Guidance

At sampling time, the model generates two predictions:

```text
ε_uncond = U-Net(x_t, t, null_class)
ε_cond   = U-Net(x_t, t, class)

ε = ε_uncond + s(ε_cond - ε_uncond)
```

where `s` is the guidance scale.

For example:

```bash
python src/sample.py --class-label 8 --guidance-scale 7.5
```

generates class `8`, which corresponds to `ship` in CIFAR-10.

## Results

Generated CIFAR-10 samples from pure Gaussian noise using the latent diffusion pipeline.

Cars generated with `guidance_scale=7.5`:

![Generated Cars](assets/gensample_vae128-unet256_car.png)

The current model is primarily an implementation and learning project focused on understanding the LDM pipeline rather than achieving state-of-the-art CIFAR-10 image quality.

## Training

### 1. Train the VAE

```bash
python src/vae_train.py
```

The trained VAE is used to create the latent representation required by the diffusion model.

### 2. Extract Latents

```bash
python src/emcode_data.py
```

This encodes the CIFAR-10 training set using the VAE and saves the normalized latents and their class labels.

### 3. Train the Latent DDPM

```bash
python src/unet_train.py
```

The U-Net learns to predict noise added to the normalized latent representations.

### 4. Generate Samples

```bash
python src/sample.py --class-label 8 --guidance-scale 7.5 --num-samples 8 --output samples/ships.png
```

CIFAR-10 class labels:

```text
0  airplane
1  automobile
2  bird
3  cat
4  deer
5  dog
6  frog
7  horse
8  ship
9  truck
```

The guidance scale and class label can be changed directly from the command line.

## Project Structure

```text
stable_diffusion/
│
├── src/
│   ├── vae_train.py
│   ├── emcode_data.py
│   ├── unet_train.py
│   ├── sample.py
│   ├── variational_autoencoder.py
│   ├── unet.py
│   ├── scheduler.py
│   └── loss.py
│
├── assets/
│   └── gensample_vae128-unet256_car.png
│
├── data/
├── models/
└── README.md
```

## Papers

* **Auto-Encoding Variational Bayes**
  Kingma & Welling

* **Denoising Diffusion Probabilistic Models**
  Ho et al.

* **High-Resolution Image Synthesis with Latent Diffusion Models**
  Rombach et al.
