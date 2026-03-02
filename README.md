<p align="center">
  <h1 align="center">TTDnet — ThingsThoughtDiffusion</h1>
  <p align="center"><em>Reconstructing Visual Perception from EEG Signals via Multi-Scale Neural Encoding and Latent Diffusion</em></p>
</p>

---
![TTDnet_arch](https://github.com/user-attachments/assets/3ec74fbf-3dd9-4a2c-a1f8-ee8bcead8bd2)
<img width="993" height="520" alt="Screenshot 2026-03-02 131629" src="https://github.com/user-attachments/assets/77a42dfc-61e2-4e3e-a3d8-ba9aa492b37b" />

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Stage A1 — Self-Supervised EEG Pre-training](#stage-a1--self-supervised-eeg-pre-training)
  - [Stage B — Generative Fine-tuning (EEG → Image)](#stage-b--generative-fine-tuning-eeg--image)
- [Latent Space Transformations](#latent-space-transformations)
- [Datasets](#datasets)
- [Training Pipeline](#training-pipeline)
- [Evaluation](#evaluation)
- [Repository Structure](#repository-structure)

---

## Overview

TTDnet is a two-stage framework for generating photorealistic images directly from electroencephalography (EEG) recordings. Given a raw multi-channel EEG epoch captured while a subject views an object, TTDnet reconstructs a semantically faithful image of that object. The system bridges the gap between low-SNR neural signals and high-fidelity image generation by chaining:

1. **A self-supervised EEG encoder** (InceptSADNet) pre-trained to learn robust temporal-spatial neural representations,
2. **An IP-Adapter–style conditioning bridge** that translates EEG latents into the semantic embedding space of a large-scale text-to-image diffusion model, and
3. **A Stable Diffusion XL (SDXL) backbone** fine-tuned with LoRA to generate 512 × 512 images conditioned on EEG rather than text.

The pipeline requires no textual descriptions at any point; visual content is decoded entirely from brain activity.

---

## Architecture

The full architecture is divided into two sequential training stages. Stage A learns a general-purpose EEG representation via masked signal reconstruction; Stage B fine-tunes the frozen encoder together with a conditioning bridge and an SDXL diffusion backbone on paired EEG–image data.

### Stage A1 — Self-Supervised EEG Pre-training

**Goal:** Learn transferable EEG representations *without* any image labels, using large-scale unlabelled EEG recordings.

#### InceptSADEncoder (`sc_mbm/incept_encoder.py`)

The encoder transforms raw multi-channel EEG into a compact sequence of latent tokens. It consists of the following major blocks:

| Block | Description | Input → Output |
|-------|-------------|----------------|
| **Multi-Scale Temporal Convolutions** | Three parallel 1-D convolution branches with kernel sizes 7, 15, and 31 time-steps extract features at short, medium, and long temporal scales simultaneously. Each branch applies `Conv2d → BatchNorm → ELU`. Outputs are concatenated along the filter dimension. | `[B, 1, C, T]` → `[B, 3·F₁, C, T]` |
| **Depthwise Spatial Convolution** | A grouped convolution with kernel `(C, 1)` collapses the spatial (channel) dimension, learning per-filter spatial weights across all EEG electrodes, followed by `BatchNorm → ELU → AvgPool → Dropout`. | `[B, 3·F₁, C, T]` → `[B, 3·F₁·D, 1, T/4]` |
| **Squeeze-and-Excitation (SE) Block** | A channel recalibration module that adaptively re-weights filter responses: `GlobalAvgPool → FC → ReLU → FC → Sigmoid → scale`. This suppresses noisy channels and amplifies informative ones. | `[B, F, 1, T']` → `[B, F, 1, T']` |
| **Projection & Tokenization** | A `1×1 Conv → BatchNorm → ELU → AvgPool → Dropout` projection maps features to the target embedding dimension, then spatially reshapes the output into a token sequence. | `[B, F, 1, T']` → `[B, seq_len, embed_dim]` |
| **Transformer Encoder** | A stack of `L` standard Transformer blocks, each with multi-head self-attention (MHSA) and a GELU feed-forward network (FFN), both wrapped in Pre-LayerNorm residual connections. | `[B, seq_len, embed_dim]` → `[B, seq_len, embed_dim]` |

Default hyperparameters: `embed_dim = 1024`, `depth (L) = 6`, `num_heads = 8`, `F₁ = 8`, `D = 2`.

#### Temporal Masking Pre-training Wrapper (`sc_mbm/incept_pretrain.py`)

Instead of the patch-level random masking used in standard MAE, we employ **contiguous temporal masking** tailored to EEG:

1. **Masking Stage:** Multiple random contiguous time segments are zeroed out in the raw EEG input (default: 50% of all time-steps). This forces the encoder to infer missing temporal context from surrounding neural activity.
2. **Encoding:** The masked signal passes through the full InceptSADEncoder, producing latent tokens.
3. **Lightweight Decoder:** A 4-layer Transformer decoder with sinusoidal positional embeddings and a linear prediction head reconstructs the original (un-masked) signal from the encoder's output.
4. **Reconstruction Loss:** MSE is computed *only* over the masked time regions, encouraging the encoder to build a holistic understanding of the temporal dynamics rather than merely copying visible segments.

The pre-training loop (`stageA1_incept_pretrain.py`) trains the encoder + decoder jointly with AdamW (lr = 4e-4, cosine schedule) for 200 epochs on large-scale unlabelled EEG data. After pre-training, only the encoder weights are transferred to Stage B; the decoder is discarded.

> **Alternative encoder path:** The codebase also retains a ViT-based Masked Autoencoder (`sc_mbm/mae_for_eeg.py`) with 1-D Patch Embedding and standard random patch masking, used with the original EEG dataset and the SD 1.5 backend. The InceptSADEncoder supersedes this for the SDXL pipeline.

---

### Stage B — Generative Fine-tuning (EEG → Image)

**Goal:** Learn to generate images that match the visual stimulus a subject was viewing, given only their EEG recording.

#### EEG Conditioning Wrapper (`dc_ldm/sdxl_pipeline.py → EEGConditioningWrapper`)

A thin wrapper that chains the frozen pre-trained InceptSADEncoder with the IP-Adapter Bridge and exposes a unified conditioning interface. On a forward pass it returns two outputs: (a) cross-attention conditioning tokens for the UNet, and (b) the raw EEG latent for CLIP alignment supervision.

#### IP-Adapter Bridge (`dc_ldm/ip_adapter_bridge.py`)

The bridge converts the EEG encoder's variable-length token sequence into a fixed-size conditioning signal compatible with SDXL's cross-attention mechanism. It has two sub-modules:

| Sub-Module | Description | Input → Output |
|------------|-------------|----------------|
| **Perceiver Resampler** | A set of 16 learnable *latent query tokens* attend to the EEG encoder tokens via iterative cross-attention (2 Resampler Layers). Each layer consists of: `LayerNorm → MultiHeadCrossAttention → Residual → LayerNorm → FFN → Residual`. The Resampler compresses the variable-length EEG sequence into a fixed set of 16 conditioning tokens projected to match SDXL's 2048-dim context space. | `[B, seq_len, 1024]` → `[B, 16, 2048]` |
| **CLIP Alignment Head** | Mean-pools the EEG latent → MLP (1024 → 1024, GELU, → 768) → L2-normalize. A learnable `logit_scale` parameter controls the temperature of the symmetric InfoNCE contrastive loss that aligns EEG embeddings with CLIP ViT-L/14 image embeddings. | `[B, seq_len, 1024]` → `[B, 768]` |

#### Diffusion Backbone — Stable Diffusion XL (`dc_ldm/sdxl_pipeline.py → EEGtoImageSDXL`)

The generative core is built on SDXL (Stable Diffusion XL Base 1.0), modified for EEG-conditioned generation:

| Component | Role | Details |
|-----------|------|---------|
| **VAE (AutoencoderKL)** | Encodes images into / decodes images from a spatial latent space. Runs in FP32 for numerical stability (its internal `exp()` overflows in FP16). | `[B, 3, 512, 512]` ↔ `[B, 4, 64, 64]` (48× compression). Frozen during training. |
| **UNet (UNet2DConditionModel) + LoRA** | The denoising network. The base UNet is frozen; LoRA adapters (rank-16, α = 16, Gaussian initialization) are injected into all `to_q`, `to_k`, `to_v`, and `to_out.0` attention projection matrices, enabling parameter-efficient fine-tuning. | Receives noisy latents + EEG conditioning tokens via cross-attention. Only LoRA weights (~20M params) are updated. |
| **DPM++ 2M Karras Scheduler** | A fast ODE-based noise scheduler that requires only 25 denoising steps (vs. 250 for PLMS/DDIM in SD 1.5), dramatically reducing inference time. | Used for both training (noise addition) and inference (iterative denoising). |
| **CLIP ViT-L/14 Image Encoder** | Extracts 768-dim image embeddings from ground-truth images during training for the contrastive CLIP alignment loss. Frozen. | Not used at inference. |
| **Pooled Conditioning Projection** | A linear layer that mean-pools the 16 cross-attention tokens and projects them to 1280-dim, replacing SDXL's pooled text embedding (`text_embeds`). | `[B, 16, 2048]` → mean → `[B, 1280]` |

#### Training Objective

The total loss during fine-tuning is a weighted sum:

```
L_total = L_diffusion + 0.5 · L_clip
```

- **L_diffusion (MSE):** Standard ε-prediction loss — the UNet predicts the noise added to the VAE latent, and the MSE between predicted and actual noise is minimized.
- **L_clip (InfoNCE):** Symmetric contrastive loss aligning the EEG-derived embedding with the CLIP image embedding in a shared 768-dim space. Computed every 4 batches to reduce CLIP encoder overhead.

#### Inference Pipeline

At inference, no ground-truth images are needed:

```
Raw EEG → InceptSADEncoder → EEG Latent Tokens
       → Perceiver Resampler → 16 Conditioning Tokens [B, 16, 2048]
       → Pooled Projection   → Pooled Embedding [B, 1280]

Random Gaussian Noise [B, 4, 64, 64]
  → 25 DPM++ Steps (UNet denoising, conditioned via cross-attention)
  → Denoised VAE Latent [B, 4, 64, 64]
  → VAE Decoder
  → Reconstructed Image [B, 3, 512, 512]
```

> **Legacy SD 1.5 path:** The codebase retains a full Stable Diffusion 1.5 pipeline (`dc_ldm/ldm_for_eeg.py → eLDM`) with DDIM/PLMS sampling (250 steps), a simpler `cond_stage_model` using linear projection, and the ViT-based `eeg_encoder`. This is used when `config.model_type = 'sd15'`.

---

## Datasets

### ThingsEEG (Primary Dataset for finetuning)

| Property | Value |
|----------|-------|
| **Source** | ThingsEEG dataset (Gifford et al.) |
| **Subjects** | Up to 50 (default: subjects 1–5) |
| **EEG channels** | 63 |
| **Sampling rate** | Resampled to 512 time-points per epoch |
| **Stimuli** | 1,854 unique object concepts from the THINGS database |
| **Split strategy** | By concept — test set contains held-out object categories to evaluate generalization to unseen visual concepts |
| **Test ratio** | 10% of concepts |
| **Image size** | 512 × 512 (for SDXL) |
| **Preprocessing** | Raw `.edf` → band-pass filter → epoch extraction → resampling → per-channel z-normalization → `.pth` files (`preprocess_things_eeg.py`) |


### Pre-training Data

| Property | Value |
|----------|-------|
| **Source** | PhysioNet / MNE datasets (`.edf` → `.npy` via `preprocess_edf_to_npy.py`) |
| **Channels** | 64 |
| **Purpose** | Self-supervised temporal masking pre-training of InceptSADEncoder (no image labels) |

---

## Training Pipeline

### Stage A1 — EEG Pre-training

```
python stageA1_incept_pretrain.py
```

| Parameter | Value |
|-----------|-------|
| Encoder | InceptSADEncoder (embed_dim=1024, depth=6, heads=8) |
| Masking | 50% contiguous temporal masking |
| Decoder | 4-layer Transformer (512-dim) |
| Optimizer | AdamW (lr=4e-4, β=(0.9, 0.95), weight_decay=0.05) |
| Epochs | 200 |
| Batch size | 256 |
| Loss | MSE (masked regions only) |
| Output | Encoder checkpoint (`checkpoint_best.pth`) |

### Stage B — Generative Fine-tuning

```
python eeg_ldm.py
```

| Parameter | Value |
|-----------|-------|
| Diffusion model | SDXL Base 1.0 |
| EEG encoder | Frozen InceptSADEncoder (from Stage A1) |
| Bridge | Perceiver Resampler (16 tokens, 2 layers) |
| UNet adaptation | LoRA (rank=16, α=16) on Q/K/V/Out projections |
| Optimizer | AdamW (lr=5.3e-5, weight_decay=0.01) |
| LR schedule | Cosine with 2-epoch linear warmup |
| Batch size | 2 × 16 gradient accumulation = 32 effective |
| Samples per epoch | 15,000 (random subset; full dataset seen across epochs) |
| Epochs | 50 |
| Precision | Mixed (AMP): VAE in FP32, UNet in FP16 |
| Loss | MSE (diffusion) + 0.5 × InfoNCE (CLIP alignment) |
| CLIP loss frequency | Every 4th batch |
| Scheduler | DPM++ 2M Karras |
| Inference steps | 25 |
| Checkpoint | Per-epoch (latest + best) with full resume support |
| Tracking | Weights & Biases |
| Training time | ~42 hours (single GPU) |

---

## Evaluation

After training, the framework generates multiple samples per test EEG epoch and evaluates them against ground-truth images using:

| Metric | Type | Measures |
|--------|------|----------|
| **MSE** | Pixel-level | Mean squared error (lower is better) |
| **PCC** | Pixel-level | Pearson correlation coefficient (higher is better) |
| **SSIM** | Structural | Structural similarity index (higher is better) |
| **LPIPS** | Perceptual | Learned perceptual similarity via AlexNet (lower is better) |
| **FID** | Distributional | Fréchet Inception Distance via InceptionV3 (lower is better) |
| **Top-1 Accuracy** | Semantic | 50-way classification accuracy using ViT-H/14 (higher is better) |

Evaluation is implemented in `eval_metrics.py` and operates in both pair-wise and n-way scoring modes.

---

## Repository Structure

```
TTDnet/
├── code/
│   ├── config.py                        # All hyperparameter configurations
│   ├── dataset.py                       # EEG pre-training & original EEG datasets
│   ├── things_dataset.py                # ThingsEEG paired dataset (EEG + images)
│   ├── eeg_ldm.py                       # Main entry point (training + generation)
│   ├── eval_metrics.py                  # MSE, PCC, SSIM, LPIPS, FID, n-way accuracy
│   │
│   ├── stageA1_eeg_pretrain.py          # Stage A1: ViT-MAE pre-training script
│   ├── stageA1_incept_pretrain.py       # Stage A1: InceptSAD pre-training script
│   │
│   ├── preprocess_edf_to_npy.py         # Raw .edf → .npy conversion
│   ├── preprocess_things_eeg.py         # ThingsEEG preprocessing pipeline
│   │
│   ├── sc_mbm/                          # Self-supervised EEG encoders
│   │   ├── InceptSADNet.py              #   InceptSADNet classification model
│   │   ├── incept_encoder.py            #   InceptSADEncoder (used in pipeline)
│   │   ├── incept_pretrain.py           #   Temporal masking pre-training wrapper
│   │   ├── mae_for_eeg.py              #   ViT-based MAE encoder (legacy)
│   │   ├── trainer.py                   #   MAE training utilities
│   │   └── utils.py                     #   Checkpoint save/load helpers
│   │
│   └── dc_ldm/                          # Diffusion & conditioning modules
│       ├── sdxl_pipeline.py             #   SDXL pipeline (train + generate)
│       ├── ip_adapter_bridge.py         #   Perceiver Resampler + CLIP alignment
│       ├── ldm_for_eeg.py              #   SD 1.5 pipeline (legacy)
│       ├── util.py                      #   Config instantiation utilities
│       │
│       ├── models/
│       │   ├── autoencoder.py           #   VAE (SD 1.5 path)
│       │   └── diffusion/
│       │       ├── ddpm.py              #   DDPM implementation
│       │       ├── ddim.py              #   DDIM sampler
│       │       ├── plms.py              #   PLMS sampler
│       │       └── classifier.py        #   Classifier-free guidance
│       │
│       └── modules/
│           ├── attention.py             #   Cross-/Self-/Linear Attention
│           ├── x_transformer.py         #   Extended Transformer blocks
│           ├── ema.py                   #   Exponential Moving Average
│           ├── diffusionmodules/        #   UNet blocks, timestep embed, utils
│           ├── encoders/               #   CLIP/FrozenEncoder wrappers
│           ├── distributions/          #   Gaussian distributions
│           └── losses/                 #   Perceptual & VQ losses
```

---
