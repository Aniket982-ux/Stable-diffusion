import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

# Distributed training
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Import local models
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion
from ddpm import DDPMSampler
from clip import CLIP
import model_loader
import pipeline

# For progress tracking
import wandb

# In-distribution captions for TinyDummyDataset (used with --use_dummy flag)
DUMMY_CAPTIONS = [
    "a ninja with red eyes and spiky hair, naruto style",
    "a shinobi standing in a forest, anime illustration",
    "a warrior with a headband, naruto anime style",
    "a young ninja with whisker marks on his cheeks",
    "a powerful ninja with glowing chakra, anime art",
]


class TinyDummyDataset(Dataset):
    """
    Very lightweight memory dataset or dummy fallback if HuggingFace/network is slow.
    Useful for quick compile verification test runs.
    """
    def __init__(self, size=16):
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        # Create dummy colorful synthetic floral pattern
        img = Image.new("RGB", (512, 512), color=(idx * 15 % 255, (255 - idx * 10) % 255, 100))
        # Add some circular features to simulate a flower shape
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.ellipse([150, 160, 360, 360], fill=(255, 200, 0), outline=(200, 50, 0))
        draw.ellipse([220, 220, 290, 290], fill=(120, 50, idx * 8 % 255))
        
        pixel_values = self.transform(img)
        caption = DUMMY_CAPTIONS[idx % len(DUMMY_CAPTIONS)]
        return {"pixel_values": pixel_values, "caption": caption}

def get_time_embedding(timesteps, device):
    """
    Generates time embeddings of size (Batch_Size, 320) for batch processing
    """
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32, device=device) / 160)
    # x: (Batch_Size, 1) @ freqs[None]: (1, 160) -> (Batch_Size, 160)
    x = timesteps.to(dtype=torch.float32, device=device)[:, None] * freqs[None]
    # concatenate sin and cos -> (Batch_Size, 320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

def main():
    parser = argparse.ArgumentParser(description="Train Stable Diffusion (UNet) on text-to-image dataset")
    parser.add_argument("--dataset_name", type=str, default="lambdalabs/naruto-blip-captions", help="HuggingFace dataset to load")
    parser.add_argument("--image_column", type=str, default="image", help="Column containing the images")
    parser.add_argument("--caption_column", type=str, default="text", help="Column containing the text captions")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split or slice to load (e.g. train[:2])")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--use_dummy", action="store_true", help="Force train on synthetic dummy data for instant testing")
    parser.add_argument("--save_interval", type=int, default=250, help="Save checkpoint every N steps (recommend 250+)")
    parser.add_argument("--eval_interval", type=int, default=10, help="Log generated evaluation samples every N steps")
    parser.add_argument("--model_file", type=str, default="data/v1-5-pruned-emaonly.ckpt", help="Path to checkpoint weights")
    parser.add_argument("--vocab_file", type=str, default="data/vocab.json", help="Path to vocabulary file")
    parser.add_argument("--merges_file", type=str, default="data/merges.txt", help="Path to merges file")
    parser.add_argument("--device", type=str, default="auto", help="Device to use for training (auto, cuda, cpu)")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="stable-diffusion-finetune", help="Weights & Biases project name")
    parser.add_argument("--wandb_name", type=str, default="naruto-finetune-run", help="Weights & Biases run name")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint file to resume training from")
    parser.add_argument("--train_from_scratch", action="store_true", help="Initialize UNet (Diffusion) weights randomly instead of loading pretrained weights, while keeping frozen CLIP and VAE preloaded.")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of linear warmup steps for training from scratch")
    
    args = parser.parse_args()

    # Detect if we are running in distributed mode
    is_distributed = "WORLD_SIZE" in os.environ or "RANK" in os.environ
    if is_distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        local_rank = 0
        global_rank = 0
        world_size = 1
        if args.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = args.device

    if global_rank == 0:
        print(f"Training Target Device: {device} (Distributed: {is_distributed}, World Size: {world_size})")

    # Initialize wandb on EVERY GPU rank so each GPU gets its own plot.
    # Runs are grouped by args.wandb_name so you see GPU-0 and GPU-1 side-by-side
    # in a single W&B Group on the dashboard.
    if not args.no_wandb:
        if global_rank == 0:
            print("--------------------------------------------------")
            print(f"Initializing Weights & Biases Logging: {args.wandb_project} (all ranks)")
            print("--------------------------------------------------")
        # If resuming, only rank 0 peeks for the previous run ID
        wandb_resume_id = None
        if args.resume and os.path.exists(args.resume) and global_rank == 0:
            _peek = torch.load(args.resume, map_location="cpu")
            wandb_resume_id = _peek.get("wandb_run_id", None)
            del _peek
        wandb.init(
            project=args.wandb_project,
            # Each GPU gets its own run: e.g. "naruto-ddp-gpu0", "naruto-ddp-gpu1"
            name=f"{args.wandb_name}-gpu{global_rank}",
            # Group ties both runs together on the W&B dashboard
            group=args.wandb_name,
            id=wandb_resume_id if global_rank == 0 else None,
            resume="allow" if (wandb_resume_id and global_rank == 0) else None,
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "use_dummy": args.use_dummy,
                "model_path": args.model_file,
                "train_from_scratch": args.train_from_scratch,
                "warmup_steps": args.warmup_steps,
                "gpu_rank": global_rank,
                "world_size": world_size,
            }
        )
    elif global_rank == 0:
        print("Weights & Biases logging is disabled.")

    # 1. Initialize tokenizer
    from transformers import CLIPTokenizer
    if global_rank == 0:
        print("Loading CLIP Tokenizer...")
    if os.path.exists(args.vocab_file) and os.path.exists(args.merges_file):
        tokenizer = CLIPTokenizer(args.vocab_file, merges_file=args.merges_file)
        if global_rank == 0:
            print("-> loaded tokenization vocab locally from data folder.")
    else:
        if global_rank == 0:
            print("Local tokenizer files not found. Falling back to downloading huggingface pre-trained tokenizer...")
        tokenizer = CLIPTokenizer.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="tokenizer")

    if global_rank == 0:
        print("Initializing components...")
    
    # Check if pre-trained weight checkpoint path exists
    if os.path.exists(args.model_file):
        if global_rank == 0:
            print(f"Pre-loaded weights found at {args.model_file}. Loading weights...")
        try:
            models = model_loader.preload_models_from_standard_weights(args.model_file, device)
            encoder = models["encoder"]
            decoder = models["decoder"]
            clip = models["clip"]
            
            if args.train_from_scratch:
                if global_rank == 0:
                    print("\n[Scratch Training] Initializing UNet (Diffusion) from absolute scratch (random weights), but using frozen pre-trained VAE and CLIP encoders!")
                diffusion = Diffusion().to(device)
            else:
                if global_rank == 0:
                    print("\n[Fine-tuning Mode] Loading pre-trained UNet (Diffusion) weights from standard checkpoint.")
                diffusion = models["diffusion"]
                
            if global_rank == 0:
                print("Successfully initialized all model parameters!")
        except Exception as e:
            if global_rank == 0:
                print(f"Error loading local checkpoint weights: {e}")
                print("Falling back to random weights initialization for sanity testing.")
            encoder = VAE_Encoder().to(device)
            decoder = VAE_Decoder().to(device)
            diffusion = Diffusion().to(device)
            clip = CLIP().to(device)
    else:
        if global_rank == 0:
            print(f"Checkpoint weight file '{args.model_file}' not found.")
            print("Auto-initializing clean model templates with random initialization (perfect for sanity test-runs).")
        encoder = VAE_Encoder().to(device)
        decoder = VAE_Decoder().to(device)
        diffusion = Diffusion().to(device)
        clip = CLIP().to(device)

    # VAE (encoder + decoder) and CLIP are fully frozen — only UNet (diffusion) trains
    encoder.eval()
    decoder.eval()
    clip.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False
    for param in clip.parameters():
        param.requires_grad = False

    # UNet (Diffusion) is the ONLY trainable component
    diffusion.train()
    for param in diffusion.parameters():
        param.requires_grad = True

    if is_distributed:
        diffusion = DDP(diffusion, device_ids=[local_rank])

    # DDPM Noise Sampler (named ddpm_sampler to avoid collision with DistributedSampler)
    # Create generator on target device for reproducibility
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    ddpm_sampler = DDPMSampler(generator)

    # 3. Load & Process Dataset
    dataloader = None
    if not args.use_dummy:
        if global_rank == 0:
            print(f"Attempting to load '{args.dataset_name}' dataset (split: '{args.dataset_split}') from HuggingFace Hub...")
        try:
            from datasets import load_dataset
            # Load the dataset
            hf_dataset = load_dataset(args.dataset_name, split=args.dataset_split)
            if global_rank == 0:
                print(f"Successfully loaded {args.dataset_name}! Sample count: {len(hf_dataset)}")
            
            # Preprocessing transforms
            resize_transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            
            class TextToImageDataset(Dataset):
                def __init__(self, data, image_col, caption_col):
                    self.data = data
                    self.image_col = image_col
                    self.caption_col = caption_col
                def __len__(self):
                    return len(self.data)
                def __getitem__(self, idx):
                    item = self.data[idx]
                    img = item[self.image_col].convert("RGB")
                    pixel_values = resize_transform(img)
                    caption = str(item[self.caption_col])
                    return {"pixel_values": pixel_values, "caption": caption}
            
            train_dataset = TextToImageDataset(hf_dataset, args.image_column, args.caption_column)
            if is_distributed:
                dist_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
                dataloader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    sampler=dist_sampler,
                    drop_last=True,
                    num_workers=2,
                    persistent_workers=True,
                    pin_memory=True
                )
            else:
                dataloader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=2,
                    persistent_workers=True,
                    pin_memory=device.startswith("cuda")
                )
        except Exception as e:
            if global_rank == 0:
                print(f"Could not load HuggingFace dataset or network issues: {e}")
                print("Falling back to TinyDummyDataset (synthetic floral generation) for training validation...")
            args.use_dummy = True

    if args.use_dummy or dataloader is None:
        if global_rank == 0:
            print("Using synthetic Dummy dataset for instant verification training loop.")
        train_dataset = TinyDummyDataset(size=16)
        if is_distributed:
            dist_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
            dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=dist_sampler, num_workers=0)
        else:
            dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 4. Optimizer and Loss (Use memory-optimized 8-bit AdamW on CUDA if bitsandbytes is installed)
    optimizer_name = "Standard PyTorch AdamW (32-bit)"
    try:
        import bitsandbytes as bnb
        if device.startswith("cuda"):
            optimizer = bnb.optim.AdamW8bit(diffusion.parameters(), lr=args.lr)
            optimizer_name = "BitsAndBytes AdamW (8-bit)"
        else:
            optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr)
    except ImportError:
        if global_rank == 0:
            print("Note: bitsandbytes not installed. Using standard PyTorch AdamW instead.")
        optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr)
    
    mse_loss = nn.MSELoss()

    if global_rank == 0:
        print(f"Initialized Optimizer: {optimizer_name}")
        print("\n=== Start of Training Testrun ===")
        print(f"Checking updates over {args.epochs} epoch(s).")
        print(f"Dataloader batch size: {args.batch_size}")
        print(f"Total batches per epoch: {len(dataloader)}")
        print("--------------------------------------------------")

    global_step = 0
    start_epoch = 0
    start_batch = 0
    # Enable amp (Automatic Mixed Precision) for memory savings on modern GPUs
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))

    model_dtype = next(diffusion.parameters()).dtype
    if global_rank == 0:
        print(f"Model weight precision dtype: {model_dtype}")

    # Resume from checkpoint if requested
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        if global_rank == 0:
            print(f"\n[Resume] Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        raw_diffusion = diffusion.module if hasattr(diffusion, 'module') else diffusion
        raw_diffusion.load_state_dict(ckpt["diffusion"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if device.startswith("cuda") and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", 0)
        start_batch = ckpt.get("batch_idx", 0) + 1  # resume from the next batch
        # If saved batch was the last in its epoch, advance to the next epoch cleanly
        if start_batch >= len(dataloader):
            start_epoch += 1
            start_batch = 0
        if global_rank == 0:
            print(f"[Resume] Restored to epoch {start_epoch + 1}, batch {start_batch}, global step {global_step}")

    # Startup cleanup of stale numbered/unmanaged checkpoints to protect local VM disk
    ckpt_dir = "checkpoints"
    if os.path.exists(ckpt_dir) and global_rank == 0:
        print("\n[Disk Space Safeguard] Cleaning up any old numbered files from prior failed sessions...")
        for filename in os.listdir(ckpt_dir):
            # Only delete intermediate step checkpoints
            if filename.startswith("diffusion_step_") and filename.endswith(".ckpt"):
                try:
                    file_path = os.path.join(ckpt_dir, filename)
                    # Do not delete the file if it is actively targetted by resume
                    if args.resume and os.path.abspath(args.resume) == os.path.abspath(file_path):
                        continue
                    os.remove(file_path)
                    print(f" -> Deleted old residual checkpoint: {file_path}")
                except Exception as cleanup_err:
                    print(f" -> Warning: Could not clean up {filename}: {cleanup_err}")

    for epoch in range(start_epoch, args.epochs):
        # Set epoch for the distributed sampler (required for proper shuffle)
        if is_distributed:
            dataloader.sampler.set_epoch(epoch)
            
        epoch_loss = 0.0
        epoch_batches = 0  # track actual batches processed (differs from len(dataloader) on resume)
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=(global_rank != 0))
        
        for batch_idx, batch in enumerate(progress_bar):
            # Skip batches already processed when resuming mid-epoch
            if epoch == start_epoch and batch_idx < start_batch:
                continue
            optimizer.zero_grad()

            # Linear learning rate warmup for training from scratch
            if args.train_from_scratch and global_step < args.warmup_steps:
                warmup_lr = args.lr * (global_step + 1) / args.warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            # Load and transfer batch items
            images = batch["pixel_values"].to(device, dtype=model_dtype)  # (Batch_Size, 3, 512, 512)
            captions = batch["caption"]
            
            # Encode captions to CLIP input context
            token_ids = tokenizer(
                captions, padding="max_length", max_length=77, return_tensors="pt"
            ).input_ids.to(device)
            # context: (Batch_Size, 77, 768)
            with torch.no_grad():
                context = clip(token_ids)
            
            # Encode images to latents using VAE Encoder
            with torch.no_grad():
                # Encoder needs a random noise matching target latent shape scaling
                latents_shape = (images.shape[0], 4, 64, 64)
                encoder_noise = torch.randn(latents_shape, device=device, dtype=model_dtype)
                # latents: (Batch_Size, 4, 64, 64)
                latents = encoder(images, encoder_noise)
                
            # Sample random noise & construct random timesteps
            # timesteps count standard: 1000
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, ddpm_sampler.num_train_timesteps, (latents.shape[0],), device=device)

            # Compute time-embeddings for batched timesteps -> (Batch_Size, 320)
            time_embedding = get_time_embedding(timesteps, device).to(dtype=model_dtype)

            # Apply the DDPM forward diffusion formula q(x_t | x_0) directly using the
            # SAME noise tensor that the model will be trained to predict.
            # IMPORTANT: We cannot use sampler.add_noise() here because that method
            # generates its own internal noise and only returns noisy_latents,
            # making it impossible to retrieve the exact noise used for the MSE target.
            alphas_cumprod = ddpm_sampler.alphas_cumprod.to(device=device, dtype=latents.dtype)
            sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
            sqrt_alpha_prod = sqrt_alpha_prod.flatten()
            while len(sqrt_alpha_prod.shape) < len(latents.shape):
                sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
            while len(sqrt_one_minus_alpha_prod.shape) < len(latents.shape):
                sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
            noisy_latents = sqrt_alpha_prod * latents + sqrt_one_minus_alpha_prod * noise
            
            # Predict noise target (using UNet) under mixed-precision autocast
            with torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
                predicted_noise = diffusion(noisy_latents, context, time_embedding)
                loss = mse_loss(predicted_noise, noise)

            # Backpropagation with gradient clipping to prevent exploding gradients
            if device.startswith("cuda"):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
                optimizer.step()

            # Tracking logs
            current_loss = loss.item()
            epoch_loss += current_loss
            epoch_batches += 1
            global_step += 1
            
            if global_rank == 0:
                progress_bar.set_postfix({"Loss": f"{current_loss:.4f}"})
            
            # Send step logs to W&B on ALL ranks — each GPU logs its own loss
            # so you get two separate loss curves in the same W&B group
            if not args.no_wandb:
                wandb.log({
                    "train/step_loss": current_loss,
                    "train/global_step": global_step,
                    "train/epoch": epoch + 1,
                    "train/gpu_rank": global_rank,
                })

            # Checkpoint Interval Trigger (only on main process)
            if global_step % args.save_interval == 0 and global_rank == 0:
                ckpt_dir = "checkpoints"
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, "diffusion_latest.ckpt")
                
                print(f"\n[Storage Checkpoint] Writing latest backup state to {ckpt_path}...")
                raw_diffusion = diffusion.module if hasattr(diffusion, 'module') else diffusion
                try:
                    torch.save({
                        "diffusion": raw_diffusion.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "wandb_run_id": wandb.run.id if not args.no_wandb else None,
                    }, ckpt_path)
                    print(f" -> Core states successfully overwritten locally.")
                except Exception as save_err:
                    print(f" -> Warning: Local checkpoint save aborted: {save_err}")

            # Image Evaluation sampling Trigger (only on main process)
            if global_step % args.eval_interval == 0 and global_rank == 0:
                print(f"\n[Eval] Running Pipeline inference sample step: {global_step}")
                
                raw_diffusion = diffusion.module if hasattr(diffusion, 'module') else diffusion
                raw_diffusion.eval()
                
                eval_prompt = "a ninja with red eyes and spiky hair, naruto style, anime illustration"
                eval_models = {
                    "clip": clip,
                    "encoder": encoder,
                    "decoder": decoder,
                    "diffusion": raw_diffusion
                }
                
                try:
                    # Run sampling under `no_grad` to output sample image
                    with torch.no_grad():
                        sampled_img_array = pipeline.generate(
                            prompt=eval_prompt,
                            uncond_prompt="",
                            do_cfg=True,
                            cfg_scale=8.0,
                            sampler_name="ddpm",
                            n_inference_steps=20, # small steps for fast verification during training
                            models=eval_models,
                            seed=42,
                            device=device,
                            tokenizer=tokenizer
                        )
                        sampled_pil = Image.fromarray(sampled_img_array)
                        
                        # Log Image to WandB
                        if not args.no_wandb:
                            wandb.log({
                                "eval/inference_sample": wandb.Image(sampled_pil, caption=f"Prompt: {eval_prompt}"),
                                "train/global_step": global_step
                            })
                        print(f"-> Successfully rendered demo sample!")
                except Exception as eval_err:
                    print(f"Eval generation test failed during running: {eval_err}")
                
                # Switch back to training mode
                diffusion.train()

        if global_rank == 0:
            average_loss = epoch_loss / max(epoch_batches, 1)
            print(f"Epoch {epoch+1} Complete | Average Loss: {average_loss:.5f}")
            if not args.no_wandb:
                wandb.log({
                    "train/epoch_average_loss": average_loss,
                    "train/epoch": epoch + 1
                })

    # Save final model weights (only on main process)
    if global_rank == 0:
        final_path = "checkpoints/diffusion_model_final.ckpt"
        os.makedirs("checkpoints", exist_ok=True)
        
        print("\n[Completed] Saving final trained model and releasing resources...")
        raw_diffusion = diffusion.module if hasattr(diffusion, 'module') else diffusion
        torch.save({
            "diffusion": raw_diffusion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "global_step": global_step,
            "epoch": args.epochs,
            "batch_idx": -1,
            "wandb_run_id": wandb.run.id if not args.no_wandb else None,
        }, final_path)
        print(f" -> Completed final model file saved to: {final_path}")
        
        # Safely purge the massive temporary local progress backup file
        latest_backup = os.path.join("checkpoints", "diffusion_latest.ckpt")
        if os.path.exists(latest_backup):
            try:
                os.remove(latest_backup)
                print(f" -> Purged temporary local update file: {latest_backup} (VRAM & Disk recovered!)")
            except Exception as cleanup_err:
                print(f" -> Warning: Could not remove temporary backup file: {cleanup_err}")

        if not args.no_wandb:
            print(" -> Registering single final checkpoint as high-value artifact on W&B...")
            artifact = wandb.Artifact(
                name="checkpoint-final",
                type="model",
                metadata={"epoch": args.epochs, "global_step": global_step}
            )
            artifact.add_file(final_path)
            wandb.log_artifact(artifact)
            # Finish wandb execution sequence
            wandb.finish()
    elif not args.no_wandb:
        # GPU 1 also needs to close its W&B run cleanly
        wandb.finish()

    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
