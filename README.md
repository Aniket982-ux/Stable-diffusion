# Stable Diffusion v1.5 Fine-Tuning & Scratch UNet Training

This repository implements a modular, memory-efficient, and distributed training pipeline for **Stable Diffusion v1.5**. It supports both fine-tuning the model from pre-trained weights and training the UNet (Diffusion) from absolute scratch (random initialization) while leveraging frozen, pre-trained VAE and CLIP weights.

Designed to run on resource-constrained multi-GPU environments—such as a dual NVIDIA Tesla T4 setup on Kaggle—the repository incorporates state-of-the-art memory optimization techniques to successfully train a ~1.07 Billion parameter pipeline within highly restrictive VRAM limitations.

---

## 🏗️ Architecture & Component Breakdown

The pipeline is split into four distinct sub-components, each carrying out a modular role in the Latent Diffusion process:

```mermaid
graph TD
    A[Text Prompt] -->|CLIP Tokenizer| B[Token IDs]
    B -->|CLIP Text Encoder (Frozen, fp16)| C[Context Embeddings (Dim: 768)]
    D[Input Image] -->|VAE Encoder (Frozen, fp16)| E[Latents (Dim: 4x64x64)]
    E -->|Forward Diffusion| F[Noisy Latents]
    F -->|Denoising UNet (Trainable, fp32)| G[Predicted Noise]
    H[DDPM Sampler] -->|Denoising Schedule| F
    G -->|Loss / Backpropagation| H
    F -->|VAE Decoder (Frozen, CPU/fp16)| I[Generated Image]
```

### Modular Parameter Profile
Below is the exact distribution and parameter footprint of the four subsystems:

| Component | Module/Class | Parameter Count | Mode during Training | Precision | Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CLIP Text Encoder** | `CLIP` | **123.06 M** | ❄️ Frozen (No Grads) | `float16` | Embeds text prompts into condition embeddings. |
| **VAE Encoder** | `VAE_Encoder` | **34.16 M** | ❄️ Frozen (No Grads) | `float16` | Compresses raw input images into latent space (8x downscaling). |
| **VAE Decoder** | `VAE_Decoder` | **49.49 M** | ❄️ Frozen (No Grads) | `float16` / `float32` | Reconstructs latent representations back to pixel space. |
| **Denoising UNet** | `Diffusion` | **859.52 M** | 🔥 Trainable | `float32` (AMP Mixed) | Iteratively predicts noise matching the target timestep. |
| **Total Pipeline** | - | **1,066,235,307** | - | - | **~1.07 Billion Parameters** |

---

## ⚡ Challenges Faced & Technical Solutions Implemented

Training or fine-tuning a 1.07 Billion parameter Stable Diffusion model on modern cloud GPUs is straightforward when unlimited VRAM is available. However, executing this on a **Dual NVIDIA Tesla T4 (15 GB VRAM per GPU)** setup under Distributed Data Parallel (DDP) poses massive memory bottlenecks. Below are the highly technical issues we encountered and how they were resolved.

### 1. Dual UNet VRAM Bottleneck (The "Ghost Copy" Bug)
> [!WARNING]
> **Symptom:** Immediate CUDA Out-of-Memory (OOM) before a single step of training could execute when using `--train_from_scratch`.

* **Root Cause:** In scratch training mode, the pipeline calls `model_loader.preload_models_from_standard_weights` to load the frozen VAE and CLIP parameters from the pre-trained `.ckpt` file. This helper method automatically loads the pre-trained UNet model weights into GPU memory as well, allocating **~3.2 GB** of VRAM. Immediately afterward, the training script initialized a second, empty UNet from scratch (`Diffusion().to(device)`). This left two massive UNets co-existing on the GPU, pushing baseline VRAM usage to **~7.05 GB** before any batch data, activation memory, or optimizer states were loaded.
* **Solution:** We modified the loading sequence to dynamically delete the pre-trained UNet, trigger Python's garbage collection, and clear the PyTorch cache *before* initializing the scratch UNet:
  ```python
  del models["diffusion"]
  del models
  import gc; gc.collect()
  torch.cuda.empty_cache()
  diffusion = Diffusion().to(device)
  ```
  This successfully recovered **~3.2 GB** of VRAM, bringing baseline GPU memory back down to an OOM-safe level of **3.22 GB**.

### 2. DDP Memory Footprint & Activation Bottlenecks
> [!CAUTION]
> **Symptom:** CUDA OOM during the backward pass (gradients calculation) when batch size was greater than 1 or when running DDP.

To fit the training loop, backward pass, and optimizer states within 15 GB of VRAM, we deployed four layers of memory optimization:
1. **Gradient Checkpointing:** Inside `diffusion.py` for the UNet, we implemented gradient checkpointing across all encoder, bottleneck, and decoder blocks. By using `torch.utils.checkpoint.checkpoint` with `use_reentrant=False`, we trade a minor computational overhead (~20% increase in backward pass duration) for an enormous reduction in VRAM activation footprint (from **~7-9 GB** down to **<100 MB**).
   To ensure PyTorch's autograd tracks gradients properly through the frozen VAE latents, we explicitly enable gradients on the input latents during the forward pass:
   ```python
   if self.gradient_checkpointing and self.training:
       x = x.clone().requires_grad_(True)
   ```
2. **DDP `gradient_as_bucket_view=True`:** Wrapped the UNet in PyTorch's `DistributedDataParallel` with `gradient_as_bucket_view=True`. This forces gradients to be views into DDP communication buckets instead of separate memory allocations, saving an additional **~3.4 GB** of VRAM per GPU.
3. **BitsAndBytes 8-bit AdamW:** Rather than using the default 32-bit AdamW optimizer (which requires 8 bytes of state per trainable parameter), we integrated `bitsandbytes` to load an 8-bit AdamW optimizer. This reduced VRAM occupied by optimizer states by **75%** (from 4 bytes to 1 byte per state parameter).
4. **Frozen Model Casting (`float16`):** The frozen VAE Encoder and CLIP models are converted to `float16` on CUDA, reducing their combined footprint on each GPU by over **1.5 GB**.

### 3. Mixed Precision Evaluation Dtype Mismatch
> [!IMPORTANT]
> **Symptom:** Evaluation failed at step 100 with the error: `expected mat1 and mat2 to have the same dtype, but got: c10::Half != float`.

* **Root Cause:** To maximize memory efficiency, the frozen CLIP and VAE Encoder models are cast to `float16`, while the UNet remains in `float32`. During the training loop, `torch.amp.autocast("cuda")` automatically manages the conversions between `float16` activations and `float32` master weights. However, at step 100, the evaluation loop triggered the image generation pipeline (`pipeline.generate`) *without* an autocast context. Consequently, `float16` prompt embeddings output by CLIP hit the `float32` linear/convolution weights of the UNet, causing a native PyTorch matrix multiplication crash.
* **Solution:** We wrapped the evaluation inference step inside [train.py](train.py) with an autocast block:
  ```python
  with torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
      sampled_img_array = pipeline.generate(...)
  ```
  This unifies the dtype context during evaluation. Since evaluation runs under `with torch.no_grad():`, it does not create a computation graph, meaning VRAM utilization remains extremely low, and the autocast execution runs with maximum VRAM and compute efficiency.

---

## 📈 Experiment Tracking & Multi-GPU Logging

The repository natively integrates with **Weights & Biases (W&B)**. 

When executing a distributed training run across multiple GPUs via `torchrun`:
* **Grouped Runs:** Runs are grouped under the same run ID and group name. Each GPU rank generates its own chart (e.g. `naruto-ddp-gpu0` and `naruto-ddp-gpu1`), allowing you to compare GPU execution metrics side-by-side.
* **Metrics Tracked:**
  - `train/step_loss`: Real-time batch MSE loss.
  - `train/epoch_average_loss`: Epoch-averaged loss.
  - `eval/inference_sample`: Generated validation images rendered using the evaluation prompt every `--eval_interval` steps.
* **Artifact Logging:** Checkpoints are logged as versioned W&B Artifacts containing model weights, optimizer states, AMP scaler states, and step metrics for seamless resumption on other nodes.

---

## 🚀 Replication & Execution Guide

Follow these steps to replicate the training run on a local workstation or a cloud notebook environment (like Kaggle or Colab).

### 📋 Prerequisites
Ensure your workspace matches the layout below:
```
Stable-diffusion/
├── data/
│   ├── v1-5-pruned-emaonly.ckpt  # Stable Diffusion v1.5 weights
│   ├── vocab.json                # Tokenizer vocab
│   └── merges.txt                # Tokenizer merges
├── train.py
├── diffusion.py
├── pipeline.py
└── requirements.txt
```

### Step-by-Step Kaggle Notebook Setup

Create a new Kaggle Notebook with **GPU T4 x2** accelerator enabled and execute the cells below:

#### Cell 1: Clone Repository & Install Dependencies
```python
# Clone the repository
!git clone https://github.com/Aniket982-ux/Stable-diffusion.git
%cd Stable-diffusion

# Install required dependencies
!pip install -q -r requirements.txt
!pip install -q bitsandbytes
```

#### Cell 2: Fetch Pre-Trained Weights
```python
import os
os.makedirs("data", exist_ok=True)

# Download tokenizer and model weights
!wget -q --show-progress -O data/v1-5-pruned-emaonly.ckpt "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt"
!wget -q -O data/vocab.json "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/tokenizer/vocab.json"
!wget -q -O data/merges.txt "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/tokenizer/merges.txt"
```

#### Cell 3: Authorize Weights & Biases
```python
import wandb
# Provide your API key when prompted
wandb.login()
```

#### Cell 4: Launch Distributed Training
Run the training script using PyTorch's distributed runner.
```bash
%%bash
torchrun --standalone --nproc_per_node=2 train.py \
    --dataset_name "lambdalabs/naruto-blip-captions" \
    --image_column "image" \
    --caption_column "text" \
    --epochs 10 \
    --batch_size 1 \
    --lr 1e-4 \
    --train_from_scratch \
    --warmup_steps 500 \
    --save_interval 250 \
    --eval_interval 100 \
    --model_file "data/v1-5-pruned-emaonly.ckpt" \
    --vocab_file "data/vocab.json" \
    --merges_file "data/merges.txt" \
    --wandb_project "sd-unet-scratch" \
    --wandb_name "naruto-ddp"
```

---

## 🛠️ Command-Line Arguments Reference

Customize training parameters by modifying the following arguments to `train.py`:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset_name` | `str` | `lambdalabs/naruto-blip-captions` | HuggingFace dataset path. |
| `--train_from_scratch` | `flag` | `False` | Initialise UNet weights randomly while keeping pre-trained VAE and CLIP weights. |
| `--warmup_steps` | `int` | `500` | Warmup steps for linear learning rate scheduler (useful for scratch training). |
| `--save_interval` | `int` | `250` | Step interval for saving resumable checkpoints. |
| `--eval_interval` | `int` | `10` | Step interval for rendering evaluation images. |
| `--resume` | `str` | `None` | Path to a checkpoint file to resume an interrupted training run. |
| `--no_gradient_checkpointing` | `flag` | `True` (enabled) | Disable gradient checkpointing (not recommended on GPUs with <24 GB VRAM). |
| `--no_wandb` | `flag` | `False` | Disable logging to Weights & Biases. |
