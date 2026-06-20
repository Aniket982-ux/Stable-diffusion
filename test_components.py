"""
Component Test Suite for Stable Diffusion Training Pipeline
Run with: python test_components.py
All tests run on CPU — safe for 4GB VRAM machines.
"""

import os
import sys
import traceback
import torch
import torch.nn as nn

# ── helpers ─────────────────────────────────────────────────────────────────
PASS  = "[PASS]"
FAIL  = "[FAIL]"
SKIP  = "[SKIP]"
SEP   = "-" * 60

results = []

def run_test(name, fn):
    print(f"\n{SEP}")
    print(f"TEST: {name}")
    print(SEP)
    try:
        fn()
        print(f"{PASS} {name}")
        results.append((name, True, None))
    except Exception as e:
        print(f"{FAIL} {name}")
        print(f"  Error: {e}")
        traceback.print_exc()
        results.append((name, False, str(e)))

DEVICE = "cpu"

# ── TEST 1: imports ──────────────────────────────────────────────────────────
def test_imports():
    print("Importing all local modules...")
    from encoder   import VAE_Encoder
    from decoder   import VAE_Decoder
    from diffusion import Diffusion
    from clip      import CLIP
    from ddpm      import DDPMSampler
    import pipeline
    import model_loader
    print("  All local module imports successful.")

    print("Importing third-party packages...")
    import torch, torchvision, numpy, tqdm, PIL, transformers, datasets, wandb
    print(f"  torch       : {torch.__version__}")
    print(f"  torchvision : {torchvision.__version__}")
    print(f"  transformers: {transformers.__version__}")
    print(f"  datasets    : {datasets.__version__}")
    print(f"  wandb       : {wandb.__version__}")

# ── TEST 2: tokenizer ────────────────────────────────────────────────────────
def test_tokenizer():
    from transformers import CLIPTokenizer
    print("Loading tokenizer from HuggingFace (stable-diffusion-v1-5/stable-diffusion-v1-5)...")
    tokenizer = CLIPTokenizer.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="tokenizer")
    print("  Tokenizer loaded.")

    sample = "a naruto character with blue eyes standing in a forest"
    ids = tokenizer.batch_encode_plus([sample], padding="max_length", max_length=77, return_tensors="pt").input_ids
    assert ids.shape == (1, 77), f"Expected (1, 77), got {ids.shape}"
    print(f"  Token ids shape: {ids.shape}  ✓")

# ── TEST 3: CLIP text encoder ────────────────────────────────────────────────
def test_clip():
    from clip import CLIP
    from transformers import CLIPTokenizer
    print("Instantiating CLIP with random weights (no checkpoint required)...")
    clip = CLIP().to(DEVICE).eval()
    total = sum(p.numel() for p in clip.parameters())
    print(f"  CLIP parameters: {total:,}")

    tokenizer = CLIPTokenizer.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="tokenizer")
    ids = tokenizer.batch_encode_plus(
        ["a naruto character"], padding="max_length", max_length=77, return_tensors="pt"
    ).input_ids.to(DEVICE)

    with torch.no_grad():
        context = clip(ids)
    assert context.shape == (1, 77, 768), f"Expected (1,77,768), got {context.shape}"
    print(f"  CLIP output shape: {context.shape}  ✓")

# ── TEST 4: VAE encoder ──────────────────────────────────────────────────────
def test_vae_encoder():
    from encoder import VAE_Encoder
    print("Instantiating VAE_Encoder with random weights...")
    encoder = VAE_Encoder().to(DEVICE).eval()
    total = sum(p.numel() for p in encoder.parameters())
    print(f"  VAE_Encoder parameters: {total:,}")

    image   = torch.randn(1, 3, 512, 512)
    noise   = torch.randn(1, 4, 64, 64)
    with torch.no_grad():
        latents = encoder(image, noise)
    assert latents.shape == (1, 4, 64, 64), f"Expected (1,4,64,64), got {latents.shape}"
    print(f"  Encoder output shape: {latents.shape}  ✓")

# ── TEST 5: VAE decoder ──────────────────────────────────────────────────────
def test_vae_decoder():
    from decoder import VAE_Decoder
    print("Instantiating VAE_Decoder with random weights...")
    decoder = VAE_Decoder().to(DEVICE).eval()
    total = sum(p.numel() for p in decoder.parameters())
    print(f"  VAE_Decoder parameters: {total:,}")

    latents = torch.randn(1, 4, 64, 64)
    with torch.no_grad():
        images = decoder(latents)
    assert images.shape == (1, 3, 512, 512), f"Expected (1,3,512,512), got {images.shape}"
    print(f"  Decoder output shape: {images.shape}  ✓")

# ── TEST 6: Diffusion (UNet) forward pass ────────────────────────────────────
def test_diffusion_unet():
    from diffusion import Diffusion
    print("Instantiating Diffusion UNet with random weights...")
    diffusion = Diffusion().to(DEVICE).eval()
    total = sum(p.numel() for p in diffusion.parameters())
    print(f"  Diffusion parameters: {total:,}")

    latent       = torch.randn(1, 4, 64, 64)
    context      = torch.randn(1, 77, 768)
    time_emb     = torch.randn(1, 320)
    with torch.no_grad():
        output = diffusion(latent, context, time_emb)
    assert output.shape == (1, 4, 64, 64), f"Expected (1,4,64,64), got {output.shape}"
    print(f"  UNet output shape: {output.shape}  ✓")

# ── TEST 7: DDPM noise schedule math ────────────────────────────────────────
def test_ddpm_noise_math():
    from ddpm import DDPMSampler
    print("Testing DDPM noise schedule and forward diffusion formula...")
    gen = torch.Generator().manual_seed(0)
    sampler = DDPMSampler(gen)

    assert sampler.betas.shape  == (1000,), "Beta schedule wrong shape"
    assert sampler.alphas_cumprod.shape == (1000,), "Alpha cumprod wrong shape"
    print("  Noise schedule shapes ✓")

    # Replicate the EXACT forward diffusion formula used in train.py
    latents   = torch.randn(2, 4, 64, 64)
    noise     = torch.randn_like(latents)
    timesteps = torch.randint(0, 1000, (2,))

    alphas_cumprod          = sampler.alphas_cumprod
    sqrt_alpha_prod         = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod         = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(latents.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
    sqrt_one_minus_alpha    = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha    = sqrt_one_minus_alpha.flatten()
    while len(sqrt_one_minus_alpha.shape) < len(latents.shape):
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

    noisy_latents = sqrt_alpha_prod * latents + sqrt_one_minus_alpha * noise
    assert noisy_latents.shape == (2, 4, 64, 64), f"Noisy latents shape wrong: {noisy_latents.shape}"
    print(f"  Forward diffusion formula output shape: {noisy_latents.shape}  ✓")
    print(f"  Noisy latents mean={noisy_latents.mean():.4f}, std={noisy_latents.std():.4f}  ✓")

# ── TEST 8: time embedding ───────────────────────────────────────────────────
def test_time_embedding():
    print("Testing batched time embedding function...")
    # Same function as in train.py
    def get_time_embedding(timesteps, device):
        freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32, device=device) / 160)
        x = timesteps.to(dtype=torch.float32, device=device)[:, None] * freqs[None]
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

    timesteps = torch.tensor([0, 250, 500, 750, 999])
    emb = get_time_embedding(timesteps, DEVICE)
    assert emb.shape == (5, 320), f"Expected (5, 320), got {emb.shape}"
    print(f"  Time embedding shape: {emb.shape}  ✓")
    assert not torch.isnan(emb).any(), "NaN detected in time embeddings!"
    print(f"  No NaNs in time embeddings  ✓")

# ── TEST 9: MSE loss gradient flow ──────────────────────────────────────────
def test_gradient_flow():
    from diffusion import Diffusion
    print("Testing backward pass and gradient flow through UNet...")
    diffusion = Diffusion().to(DEVICE).train()

    noisy_latents = torch.randn(1, 4, 64, 64)
    context       = torch.randn(1, 77, 768)
    time_emb      = torch.randn(1, 320)
    target_noise  = torch.randn(1, 4, 64, 64)

    pred = diffusion(noisy_latents, context, time_emb)
    loss = nn.MSELoss()(pred, target_noise)
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in diffusion.parameters() if p.requires_grad)
    assert has_grad, "No gradients flowed through UNet!"
    assert not torch.isnan(loss), f"Loss is NaN!"
    print(f"  Loss value: {loss.item():.6f}  ✓")
    print(f"  Gradients non-zero in UNet  ✓")

# ── TEST 10: gradient clipping ───────────────────────────────────────────────
def test_gradient_clipping():
    from diffusion import Diffusion
    print("Testing gradient clipping correctness...")
    diffusion = Diffusion().to(DEVICE).train()

    noisy_latents = torch.randn(1, 4, 64, 64)
    context       = torch.randn(1, 77, 768)
    time_emb      = torch.randn(1, 320)
    target_noise  = torch.randn(1, 4, 64, 64)

    pred = diffusion(noisy_latents, context, time_emb)
    loss = nn.MSELoss()(pred, target_noise)
    loss.backward()

    grad_norm_before = torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
    print(f"  Grad norm before clipping: {grad_norm_before:.4f}")
    grad_norm_after = sum(
        p.grad.norm().item() ** 2 for p in diffusion.parameters() if p.grad is not None
    ) ** 0.5
    assert grad_norm_after <= 1.0 + 1e-4, f"Grad norm not clipped: {grad_norm_after:.4f}"
    print(f"  Grad norm after  clipping: {grad_norm_after:.4f}  ✓")

# ── TEST 11: HuggingFace dataset ──────────────────────────────────────────────
def test_dataset():
    from datasets import load_dataset
    from torchvision import transforms
    print("Downloading naruto-blip-captions (first 4 samples)...")
    ds = load_dataset("lambdalabs/naruto-blip-captions", split="train[:4]")
    print(f"  Loaded {len(ds)} samples  ✓")
    assert "image" in ds.column_names, "Missing 'image' column"
    assert "text"  in ds.column_names, "Missing 'text' column"
    print(f"  Columns present: {ds.column_names}  ✓")

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    sample = ds[0]
    pixel_values = transform(sample["image"].convert("RGB"))
    assert pixel_values.shape == (3, 512, 512), f"Expected (3,512,512), got {pixel_values.shape}"
    caption = str(sample["text"])
    assert len(caption) > 0, "Caption is empty"
    print(f"  Pixel values shape: {pixel_values.shape}  ✓")
    print(f"  Sample caption: '{caption[:80]}'  ✓")

# ── TEST 12: full single mini-batch end-to-end ──────────────────────────────
def test_end_to_end_minibatch():
    from encoder   import VAE_Encoder
    from decoder   import VAE_Decoder
    from diffusion import Diffusion
    from clip      import CLIP
    from ddpm      import DDPMSampler
    from transformers import CLIPTokenizer

    print("Running full end-to-end mini-batch (batch=1, CPU, random weights)...")

    tokenizer = CLIPTokenizer.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="tokenizer")
    encoder   = VAE_Encoder().to(DEVICE).eval()
    decoder   = VAE_Decoder().to(DEVICE).eval()
    clip_m    = CLIP().to(DEVICE).eval()
    diffusion = Diffusion().to(DEVICE).train()
    gen       = torch.Generator().manual_seed(42)
    sampler   = DDPMSampler(gen)

    for p in encoder.parameters():  p.requires_grad = False
    for p in clip_m.parameters():   p.requires_grad = False

    # Fake image + caption
    image   = torch.randn(1, 3, 512, 512)
    caption = ["a naruto character with spiky blond hair"]

    # Tokenize
    ids = tokenizer.batch_encode_plus(caption, padding="max_length", max_length=77, return_tensors="pt").input_ids
    with torch.no_grad():
        context = clip_m(ids)

    # Encode to latents
    enc_noise = torch.randn(1, 4, 64, 64)
    with torch.no_grad():
        latents = encoder(image, enc_noise)
    print(f"  Latents shape         : {latents.shape}")

    # Forward diffusion (exact formula from train.py)
    noise     = torch.randn_like(latents)
    timesteps = torch.randint(0, 1000, (1,))
    alphas_cumprod     = sampler.alphas_cumprod
    sqrt_a             = (alphas_cumprod[timesteps] ** 0.5).flatten().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    sqrt_one_minus_a   = ((1 - alphas_cumprod[timesteps]) ** 0.5).flatten().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    noisy_latents      = sqrt_a * latents + sqrt_one_minus_a * noise

    # UNet prediction
    def get_time_embedding(ts, dev):
        freqs = torch.pow(10000, -torch.arange(0, 160, dtype=torch.float32, device=dev) / 160)
        x = ts.to(dtype=torch.float32, device=dev)[:, None] * freqs[None]
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

    time_emb = get_time_embedding(timesteps, DEVICE)
    pred_noise = diffusion(noisy_latents, context, time_emb)
    print(f"  Predicted noise shape : {pred_noise.shape}")

    # Loss + backward
    loss = nn.MSELoss()(pred_noise, noise)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)

    assert not torch.isnan(loss), "NaN loss in end-to-end test!"
    assert pred_noise.shape == (1, 4, 64, 64)
    print(f"  MSE Loss              : {loss.item():.6f}")
    print(f"  Backward pass         : ✓")
    print(f"  Full mini-batch end-to-end: ✓")

# ── run all ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  STABLE DIFFUSION PIPELINE TEST SUITE")
    print("  Device: CPU  |  Safe for 4GB VRAM machines")
    print("=" * 60)

    run_test("1. Module & package imports",          test_imports)
    run_test("2. CLIP tokenizer",                    test_tokenizer)
    run_test("3. CLIP text encoder forward",         test_clip)
    run_test("4. VAE Encoder forward",               test_vae_encoder)
    run_test("5. VAE Decoder forward",               test_vae_decoder)
    run_test("6. Diffusion UNet forward",            test_diffusion_unet)
    run_test("7. DDPM noise schedule math",          test_ddpm_noise_math)
    run_test("8. Time embedding function",           test_time_embedding)
    run_test("9. Gradient flow through UNet",        test_gradient_flow)
    run_test("10. Gradient clipping",                test_gradient_clipping)
    run_test("11. HuggingFace naruto dataset",       test_dataset)
    run_test("12. Full end-to-end mini-batch",       test_end_to_end_minibatch)

    # ── summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - passed
    for name, ok, err in results:
        status = PASS if ok else FAIL
        print(f"  {status}  {name}")
        if err:
            print(f"         └─ {err[:100]}")
    print(f"\n  {passed}/{len(results)} tests passed  |  {failed} failed")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
