
## Stable Diffusion Model for Text to Image and Image to Image Conversion

### Introduction
The stable diffusion model is a powerful framework for generating high-quality images from text descriptions or transforming existing images into new ones. It leverages a combination of techniques including attention mechanisms, deep generative models, and the Contrastive Language-Image Pre-training (CLIP) model to achieve impressive results.

### Files Overview
1. **add_noise.ipynb**: This notebook  contains code for adding noise to images, which is a crucial step in the diffusion process to generate diverse samples.
   
2. **attention.py**: This file implements attention mechanisms, which are essential for capturing long-range dependencies and improving the quality of generated images by focusing on relevant parts of the input.

3. **clip.py**: CLIP (Contrastive Language-Image Pre-training) is a neural network that learns to understand text and images by contrasting them in a joint embedding space. This module provides functions to integrate CLIP into the diffusion model for conditioning.

4. **ddpm.py**: DDPM (Diffusion Probabilistic Models) is a type of generative model used in the diffusion process. It models the conditional distribution of pixels given the noise process, allowing for efficient generation of high-fidelity images.

5. **decoder.py**: The decoder module is responsible for generating images from latent representations produced by the diffusion model. It transforms the noise process into realistic images based on the learned distribution.

6. **demo.ipynb**: This notebook  contains a demonstration of the text-to-image or image-to-image conversion process using the stable diffusion model. It may include example code and visualizations to showcase the model's capabilities.

7. **diffusion.py**: This file  contains the core implementation of the diffusion model, including functions for sampling from the diffusion process, calculating log-likelihoods, and training the model.

8. **encoder.py**: The encoder module encodes images into latent representations, which can then be used by the decoder to generate reconstructions or transformations. It may utilize convolutional neural networks (CNNs) or other feature extractors.

9. **model_converter.py**: This script provides utilities for converting models between different formats or frameworks, facilitating interoperability and deployment in various environments.

10. **model_loader.py**: The model loader module handles loading pre-trained models and initializing them for inference or fine-tuning. It ensures consistency and reproducibility across different experiments or deployments.

11. **pipeline.py**: The pipeline module orchestrates the entire text-to-image or image-to-image conversion process, from input preprocessing to output generation. It may include functions for data loading, conditioning, sampling, and evaluation.

    
## Setup, Model Downloads and Cloud Training

To run training or inference, you must download pre-trained weights and tokenizer configuration files:

### 1. Local Downloads Setup
Create a directory named `data` in the project root folder. Then, retrieve and store the following resources:
- **Tokenizer config files**: Download `vocab.json` and `merges.txt` from the [stable-diffusion-v1-5 tokenizer directory](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/tokenizer) and place them in the `data` directory.
- **Model weights**: Download `v1-5-pruned-emaonly.ckpt` from the [stable-diffusion-v1-5 repository](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main) and place it in the `data` directory.

> **Note:** The old `runwayml/stable-diffusion-v1-5` repo has been archived. The canonical location is now `stable-diffusion-v1-5/stable-diffusion-v1-5`.

### 2. Training Dataset
The training pipeline uses the **`lambdalabs/naruto-blip-captions`** dataset — 1,221 image-caption pairs with the exact `image` and `text` schema, the canonical benchmark for fine-tuning Stable Diffusion v1.5.

The dataset is **streamed and cached automatically** by HuggingFace on the first run (~700 MB total). No manual download is required. On subsequent runs the local cache is reused instantly.

**Local cache location (Windows):**
```
C:\Users\<your-username>\.cache\huggingface\datasets\lambdalabs___naruto-blip-captions\
```

**Cloud cache location (Linux VM):**
```
~/.cache/huggingface/datasets/lambdalabs___naruto-blip-captions/
```

To use a different dataset, pass `--dataset_name` and optionally `--image_column` / `--caption_column` at the command line. Any HuggingFace image-text dataset with compatible columns is supported without code changes.

### 3. Cloud Training Framework
Fine-tuning generative diffusion models benefits heavily from high-throughput hardware. We recommend the following for a 2-epoch smoke test:

| Instance | GPU | VRAM | Spot Price | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Standard_NVads_A10_v5** *(recommended)* | NVIDIA A10 | 24 GB | ~$0.17/hr | Ampere BF16 tensor cores, OOM-safe at batch=2 |
| Standard_NC16as_T4_v3 | NVIDIA T4 | 16 GB | ~$0.20/hr | Fallback — use over NC8 for more CPU cores |
| Standard_NC24ads_A100_v4 | NVIDIA A100 | 80 GB | ~$0.80/hr | Fastest but quota is harder to obtain |

**Cloud Setup Steps:**
  1. Provision a GPU VM (spot pricing recommended for smoke tests).
  2. Clone this repository and install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
  3. Authenticate Weights & Biases (required before the first training run — skip with `--no_wandb` to disable logging entirely):
     ```bash
     wandb login   # paste your API key from wandb.ai/authorize
     ```
  4. Place `v1-5-pruned-emaonly.ckpt` (and optionally tokenizer files) in a `data/` folder, or let the script auto-initialize with random weights for pipeline verification.
  5. Run the component test suite to verify every subsystem before committing GPU time:
     ```bash
     python test_components.py
     ```
     This script independently tests: tokenizer encoding, CLIP forward pass, VAE encode/decode loop, UNet denoising forward pass, DDPM noise schedule math, time embedding generation, gradient flow and clipping through the UNet, HuggingFace dataset download and schema, and a full end-to-end mini-batch.
  6. Launch training:
     ```bash
     python train.py --epochs 2 --batch_size 2 --wandb_project stable-diffusion-finetune
     ```
  7. For a fast offline pipeline-only check (no GPU, no dataset download, no W&B):
     ```bash
     python train.py --use_dummy --device cpu --epochs 1 --batch_size 1 --no_wandb
     ```
  8. To resume a training run that was interrupted, pass the checkpoint path:
     ```bash
     python train.py --resume checkpoints/diffusion_step_500.ckpt --epochs 2
     ```
     The script restores the UNet weights, optimizer momentum, AMP scaler state, global step counter, and W&B run ID so charts continue on the same run without restarting from step 1.

---

## Model Parameter Counts & Distribution

Your custom Stable Diffusion implementation comprises four core sub-components. Below is the exact distribution and parameters profile for each modular structure:

| Component | Class Name | Exact Parameter Count | Approximate Count | Description |
| :--- | :--- | :--- | :--- | :--- |
| **VAE Encoder** | `VAE_Encoder` | $34,163,664$ | ~**$34.2$ Million** | Encodes high-dimensional images into latent space |
| **VAE Decoder** | `VAE_Decoder` | $49,490,199$ | ~**$49.5$ Million** | Reconstructs latent representations back to pixel space |
| **CLIP Text Encoder** | `CLIP` | $123,060,480$ | ~**$123.1$ Million** | Embeds your descriptive prompt inputs |
| **Diffusion (UNet)** | `Diffusion` | $859,520,964$ | ~**$859.5$ Million** | Iteratively denoises the latent representation |
| **Total Pipeline** | - | **$1,066,235,307$** | **~$1.07$ Billion** | Entire stable diffusion parameter scale |

---

## Experiment Tracking & Monitored Metrics

Our training pipeline in [train.py](train.py) uses Weights & Biases (W&B) to monitor performance in real time. The model tracks and logs the following metrics:

### 1. Training Metrics
- **`train/step_loss`**: The Mean Squared Error (MSE) computed at each optimization step. Represents the error between the noise sampled and the noise predicted by the UNet.
- **`train/epoch_average_loss`**: Mean loss across all batches actually processed in the epoch. Correctly accounts for partially-processed epochs on resumed runs.
- **`train/global_step`**: Cumulative optimization steps completed — preserved across resumes so the W&B chart is continuous.
- **`train/epoch`**: Active training epoch index.

### 2. Generative Quality Metrics
- **`eval/inference_sample`**: Every `--eval_interval` steps, [pipeline.py](pipeline.py) generates a fresh image using the in-distribution prompt `"a ninja with red eyes and spiky hair, naruto style, anime illustration"` and logs it to W&B so visual quality can be tracked over time.
- **`CLIP Score`**: Cosine similarity between text and image embeddings — measures prompt conformity (higher is better).
- **`FID`** (Fréchet Inception Distance): Distance between generated and real image distributions — measures overall visual fidelity (lower is better).

### 3. Checkpoint Artifacts
Every checkpoint saved by `--save_interval` (default every 10 steps) and the final model are logged to W&B as versioned **Artifacts** under type `model`. Each artifact carries metadata: `epoch`, `batch_idx`, and `global_step`. Artifacts can be downloaded directly from the W&B UI to resume training on a new machine without any file transfer.

**Checkpoint contents (resumable):**
```python
{
    "diffusion":    <UNet weights>,
    "optimizer":    <AdamW momentum & second moments>,
    "scaler":       <AMP loss scale factor>,
    "global_step":  <int>,
    "epoch":        <int>,
    "batch_idx":    <int>,
    "wandb_run_id": <str>   # used to resume the same W&B run
}
```


