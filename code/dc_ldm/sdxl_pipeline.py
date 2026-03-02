"""
SDXL-based EEG-to-Image pipeline using HuggingFace diffusers.

Replaces the old eLDM class for SDXL generation.
Uses DPM++ 2M Karras scheduler (25 steps vs 250 PLMS).

Training flow:
    EEG → InceptSADEncoder → IPAdapterBridge → UNet cross-attention
    Image → VAE encode → add noise → UNet predicts noise → loss

Generation flow:
    EEG → InceptSADEncoder → IPAdapterBridge → conditioning
    Random noise → denoise with DPM++ → VAE decode → image
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
)
import peft
from peft import LoraConfig
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from dc_ldm.ip_adapter_bridge import IPAdapterBridge


from diffusers.models.attention_processor import AttnProcessor

class PEFTCompatibleAttnProcessor(AttnProcessor):
    """
    Diffusers 0.21.4 passes `scale=scale` kwargs to Linear layers.
    PEFT 0.4.0 Linear models do not accept `scale` and crash.
    This processor explicitly removes `scale` before passing inputs to the layers.
    It inherits from `AttnProcessor` to remain compatible with PyTorch 1.12.1.
    """
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # Drop the scale kwarg entirely for compatibility with PEFT
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Drop the scale kwarg entirely for compatibility with PEFT
        # to_out can be a tuple or a single layer depending on the specific block type
        if type(attn.to_out) == torch.nn.modules.container.ModuleList:
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
        else:
            hidden_states = attn.to_out(hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        # Re-scale back at the end as expected
        hidden_states = hidden_states / scale

        return hidden_states


class EEGConditioningWrapper(nn.Module):
    """
    Wraps the EEG encoder + IP-Adapter bridge into a single conditioning module.
    
    This replaces cond_stage_model for SDXL.
    """
    def __init__(self, eeg_encoder, pretrained_weights=None,
                 context_dim=2048, num_tokens=16, clip_dim=768,
                 use_clip_loss=True):
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.bridge = IPAdapterBridge(
            eeg_dim=eeg_encoder.embed_dim,
            context_dim=context_dim,
            num_tokens=num_tokens,
            clip_dim=clip_dim,
            use_clip_loss=use_clip_loss,
        )
        
        # Load pre-trained encoder weights if provided
        if pretrained_weights is not None:
            self.eeg_encoder.load_checkpoint(pretrained_weights)
            print("Loaded pre-trained EEG encoder weights.")
        
        # Expose attributes for compatibility
        self.fmri_latent_dim = eeg_encoder.embed_dim
        self.fmri_seq_len = eeg_encoder.num_patches
    
    def forward(self, x):
        """
        x: [B, channels, time_len]
        returns: (cross_attn_cond, raw_latent)
            cross_attn_cond: [B, num_tokens, context_dim] for UNet
            raw_latent: [B, seq_len, embed_dim] for CLIP loss
        """
        eeg_latent = self.eeg_encoder(x)       # [B, seq_len, embed_dim]
        cond = self.bridge(eeg_latent)          # [B, num_tokens, context_dim]
        return cond, eeg_latent
    
    def get_clip_loss(self, raw_latent, image_embeds):
        return self.bridge.get_clip_loss(raw_latent, image_embeds)


class EEGtoImageSDXL:
    """
    Main pipeline class: replaces eLDM for SDXL-based generation.
    
    Handles:
    - Loading SDXL components (UNet, VAE, scheduler)
    - Fine-tuning with EEG conditioning
    - Generation with DPM++ scheduler
    """
    
    def __init__(self, eeg_encoder, device=torch.device('cpu'),
                 pretrain_root='../pretrains/',
                 sdxl_model_id='stabilityai/stable-diffusion-xl-base-1.0',
                 ddim_steps=25, global_pool=False,
                 use_clip_loss=True, pretrained_encoder_weights=None,
                 logger=None):
        
        self.device = device
        self.ddim_steps = ddim_steps
        
        # ── Load SDXL Components ──
        print("Loading SDXL VAE...")
        # SDXL VAE must run in FP32 — its internal exp() overflows in FP16
        # causing NaN latents. This is a well-known issue.
        self.vae = AutoencoderKL.from_pretrained(
            sdxl_model_id, subfolder="vae", torch_dtype=torch.float32,
            cache_dir=pretrain_root
        ).to(device)
        self.vae.requires_grad_(False)
        self.vae.eval()
        
        print("Loading SDXL UNet...")
        self.unet = UNet2DConditionModel.from_pretrained(
            sdxl_model_id, subfolder="unet", torch_dtype=torch.float16,
            cache_dir=pretrain_root
        ).to(device)
        
        # ── Scheduler: DPM++ 2M Karras ──
        print("Setting up DPM++ 2M Karras scheduler...")
        self.scheduler = DPMSolverMultistepScheduler.from_pretrained(
            sdxl_model_id, subfolder="scheduler",
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
            cache_dir=pretrain_root
        )
        
        # ── CLIP image encoder for alignment loss ──
        if use_clip_loss:
            print("Loading CLIP image encoder...")
            self.clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                "openai/clip-vit-large-patch14",
                cache_dir=pretrain_root
            ).to(device)
            self.clip_image_encoder.requires_grad_(False)
            self.clip_image_encoder.eval()
            self.clip_processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14",
                cache_dir=pretrain_root
            )
        else:
            self.clip_image_encoder = None
            self.clip_processor = None
        
        # Freeze UNet base weights
        # Only the conditioning model (encoder + bridge) + LoRA will be trained
        self.unet.requires_grad_(False)
        self.unet.eval()
        
        # SDXL context_dim = 2048 (concatenation of two CLIP encoders)
        cross_attn_dim = self.unet.config.cross_attention_dim  # 2048 for SDXL
        
        # ── Inject LoRA into UNet ──
        print("Injecting LoRA adapters into UNet...")
        unet_lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        # Use PEFT native method since diffusers 0.21.4 lacks add_adapter()
        self.unet = peft.get_peft_model(self.unet, unet_lora_config)
        self.unet.train() # Make sure LoRA layers are in train mode
        
        # ── Fix Diffusers 0.21.4 / PEFT 0.4.0 Incompatibility ──
        # Diffusers passes `scale=scale` to Linear layers, but PEFT's wrapped 
        # Linear layers don't accept it, causing a TypeError during forward pass.
        # We replace the default AttnProcessor with our patched version.
        print("Applying PEFT-compatible Attention Processor...")
        from diffusers.models.attention_processor import AttnProcessor
        attn_processors = {}
        for name in self.unet.attn_processors.keys():
            attn_processors[name] = PEFTCompatibleAttnProcessor()
        self.unet.set_attn_processor(attn_processors)
        
        # ── EEG Conditioning Module ──
        self.cond_model = EEGConditioningWrapper(
            eeg_encoder=eeg_encoder,
            pretrained_weights=pretrained_encoder_weights,
            context_dim=cross_attn_dim,
            num_tokens=16,
            clip_dim=768,
            use_clip_loss=use_clip_loss,
        ).to(device)
        
        # Pooling projection for SDXL's required text_embeds [B, 1280]
        # Initialized with small weights to prevent FP16 overflow
        self._pool_proj = nn.Linear(cross_attn_dim, 1280).to(device)
        nn.init.normal_(self._pool_proj.weight, std=0.02)
        nn.init.zeros_(self._pool_proj.bias)
        
        # VAE scaling factor
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        
        self.use_clip_loss = use_clip_loss
        print(f"EEGtoImageSDXL initialized. UNet cross_attn_dim={cross_attn_dim}")
        trainable = sum(p.numel() for p in self.cond_model.parameters() if p.requires_grad)
        trainable += sum(p.numel() for p in self._pool_proj.parameters())
        trainable += sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.unet.parameters() if not p.requires_grad)
        print(f"  Trainable params: {trainable/1e6:.1f}M  |  Frozen UNet: {frozen/1e6:.1f}M")

    # ── Training ─────────────────────────────
    def finetune(self, dataset, test_dataset, config, output_path):
        """
        Fine-tune the UNet LoRA + bridge + EEG encoder on paired EEG-image data.
        
        Features:
        - Gradient accumulation for large effective batch size
        - Cosine LR scheduler with linear warmup
        - WandB experiment tracking
        - Per-epoch checkpoint with resume capability
        """
        import wandb
        import math
        from torch.cuda.amp import autocast, GradScaler
        
        os.makedirs(output_path, exist_ok=True)
        
        bs = config.batch_size
        lr = config.lr
        num_epochs = config.num_epoch
        grad_accum_steps = getattr(config, 'gradient_accumulation_steps', 1)
        effective_bs = bs * grad_accum_steps
        samples_per_epoch = getattr(config, 'samples_per_epoch', None)
        clip_loss_every_n = 4  # Compute CLIP loss every N batches (saves ~75% CLIP overhead)
        
        # Epoch subsampling: use a random subset each epoch for speed
        # Different random subset every epoch → all data seen over time
        from torch.utils.data import RandomSampler
        # num_workers=0 on Windows avoids multiprocessing overhead that
        # can actually stall the GPU between batches.
        if samples_per_epoch and samples_per_epoch < len(dataset):
            sampler = RandomSampler(dataset, replacement=False, 
                                    num_samples=samples_per_epoch)
            dataloader = DataLoader(dataset, batch_size=bs, sampler=sampler,
                                    num_workers=0, pin_memory=True)
        else:
            dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, 
                                    num_workers=0, pin_memory=True)
        
        # Freeze VAE + UNet base, train cond_model + pool_proj + UNet LoRA
        self.vae.requires_grad_(False)
        self.unet.train()  # LoRA layers need train mode
        self.cond_model.train()
        self._pool_proj.train()
        
        # Optimizer: conditioning model + pool projection + UNet LoRA params
        trainable_params = list(self.cond_model.parameters()) + list(self._pool_proj.parameters())
        trainable_params += [p for p in self.unet.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
        scaler = GradScaler()  # AMP gradient scaler for mixed-precision training
        
        # Cosine LR scheduler with linear warmup
        warmup_epochs = 2
        steps_per_epoch = len(dataloader) // grad_accum_steps
        total_steps = steps_per_epoch * num_epochs
        warmup_steps = steps_per_epoch * warmup_epochs
        
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / max(warmup_steps, 1)
            progress = float(step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # ── Resume from checkpoint ──
        start_epoch = 0
        global_step = 0
        best_loss = float('inf')
        resume_path = getattr(config, 'checkpoint_path', None)
        
        if resume_path is not None and os.path.exists(resume_path):
            print(f"Resuming from checkpoint: {resume_path}")
            ckpt = torch.load(resume_path, map_location='cpu')
            self.cond_model.load_state_dict(ckpt['cond_model'])
            self._pool_proj.load_state_dict(ckpt['pool_proj'])
            if 'unet_lora' in ckpt:
                peft.set_peft_model_state_dict(self.unet, ckpt['unet_lora'])
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            if 'scheduler' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler'])
            if 'scaler' in ckpt:
                scaler.load_state_dict(ckpt['scaler'])
            start_epoch = ckpt.get('epoch', -1) + 1
            global_step = ckpt.get('global_step', 0)
            best_loss = ckpt.get('best_loss', float('inf'))
            print(f"  Resumed at epoch {start_epoch}, global_step {global_step}")
        
        # ── WandB init ──
        wandb_project = getattr(config, 'wandb_project', 'eeg-to-image-sdxl')
        wandb_run_name = getattr(config, 'wandb_run_name', None)
        wandb_id = None
        if resume_path and os.path.exists(resume_path):
            ckpt_data = torch.load(resume_path, map_location='cpu')
            wandb_id = ckpt_data.get('wandb_id', None)
        
        trainable_count = sum(p.numel() for p in trainable_params)
        use_wandb = True
        try:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                id=wandb_id,
                resume="allow" if wandb_id else None,
                config={
                    "batch_size": bs,
                    "effective_batch_size": effective_bs,
                    "lr": lr,
                    "num_epochs": num_epochs,
                    "grad_accum_steps": grad_accum_steps,
                    "trainable_params": trainable_count,
                    "dataset_size": len(dataset),
                    "model": "SDXL + LoRA + InceptSADEncoder",
                }
            )
        except Exception as e:
            print(f"[WARNING] WandB init failed: {e}")
            print("  Training will continue WITHOUT WandB logging.")
            print("  To fix: run 'wandb login YOUR_API_KEY' in terminal first.")
            use_wandb = False
        
        print(f"\n{'='*60}")
        print(f"Starting SDXL Fine-tuning")
        print(f"  Epochs     : {num_epochs} (starting from {start_epoch})")
        print(f"  Batch size : {bs} x {grad_accum_steps} accum = {effective_bs} effective")
        print(f"  Samples/ep : {len(dataloader)*bs} / {len(dataset)} total")
        print(f"  LR         : {lr} (cosine schedule, {warmup_epochs} warmup epochs)")
        print(f"  AMP        : enabled")
        print(f"  CLIP loss  : {self.use_clip_loss} (every {clip_loss_every_n} batches)")
        print(f"  Output     : {output_path}")
        print(f"  Trainable  : {trainable_count/1e6:.1f}M params")
        print(f"  WandB      : {wandb.run.url if (use_wandb and wandb.run) else 'disabled'}")
        print(f"{'='*60}\n")
        
        for epoch in range(start_epoch, num_epochs):
            epoch_loss = 0.0
            epoch_clip_loss = 0.0
            num_batches = 0
            nan_batches = 0
            
            optimizer.zero_grad(set_to_none=True)  # Faster than zeroing
            
            for batch_idx, batch in enumerate(dataloader):
                eeg = batch['eeg'].to(self.device, non_blocking=True)
                images = batch['image'].to(self.device, non_blocking=True)  # [B, C, H, W] in [-1, 1]
                
                if images.ndim == 4 and images.shape[1] != 3:
                    images = rearrange(images, 'b h w c -> b c h w')
                
                # VAE runs in FP32 for numerical stability
                with torch.no_grad():
                    latents = self.vae.encode(images.float()).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor
                    latents = latents.half()  # Cast to FP16 for UNet
                
                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, self.scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=self.device
                ).long()
                
                noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
                
                # ── AMP autocast for UNet forward + loss ──
                with autocast():
                    # Get EEG conditioning
                    cond, raw_eeg_latent = self.cond_model(eeg)
                    
                    # FP16-safe clamping
                    FP16_MAX = 65000.0
                    cond_safe = torch.clamp(cond, -FP16_MAX, FP16_MAX)
                    
                    added_cond_kwargs = self._get_added_cond_kwargs(
                        latents.shape[0], latents.shape[-2], latents.shape[-1],
                        cond_safe
                    )
                    
                    # UNet forward pass
                    noise_pred = self.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=cond_safe.half(),
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample
                    
                    diff_loss = F.mse_loss(noise_pred.float(), noise.float())
                    
                    # CLIP loss only every N batches to save compute
                    clip_loss = torch.tensor(0.0, device=self.device)
                    if self.use_clip_loss and self.clip_image_encoder is not None \
                            and batch_idx % clip_loss_every_n == 0:
                        with torch.no_grad():
                            clip_input = F.interpolate(
                                (images + 1) / 2,
                                size=(224, 224), mode='bilinear', align_corners=False
                            )
                            clip_embeds = self.clip_image_encoder(clip_input.half()).image_embeds
                        clip_loss = self.cond_model.get_clip_loss(raw_eeg_latent, clip_embeds.float())
                    
                    total_loss = (diff_loss + 0.5 * clip_loss) / grad_accum_steps
                
                # NaN safety
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    nan_batches += 1
                    continue
                
                scaler.scale(total_loss).backward()
                
                # Step optimizer every grad_accum_steps
                if (batch_idx + 1) % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                
                # Track un-scaled loss for logging
                real_diff = diff_loss.item()
                real_clip = clip_loss.item()
                epoch_loss += real_diff
                epoch_clip_loss += real_clip
                num_batches += 1
                
                # Log to WandB every 10 batches
                if batch_idx % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                          f"diff={real_diff:.4f} clip={real_clip:.4f} "
                          f"lr={current_lr:.2e}")
                    if use_wandb:
                        wandb.log({
                            "train/diff_loss": real_diff,
                            "train/clip_loss": real_clip,
                            "train/total_loss": real_diff + 0.5 * real_clip,
                            "train/lr": current_lr,
                            "train/epoch": epoch,
                            "train/global_step": global_step,
                        }, step=global_step)
            
            # ── Epoch summary ──
            avg_loss = epoch_loss / max(num_batches, 1)
            avg_clip = epoch_clip_loss / max(num_batches, 1)
            print(f"Epoch {epoch}/{num_epochs-1}  "
                  f"avg_diff={avg_loss:.4f}  avg_clip={avg_clip:.4f}  "
                  f"nan_skipped={nan_batches}")
            
            if use_wandb:
                wandb.log({
                    "epoch/avg_diff_loss": avg_loss,
                    "epoch/avg_clip_loss": avg_clip,
                    "epoch/nan_batches": nan_batches,
                    "epoch/epoch": epoch,
                }, step=global_step)
            
            # ── Save checkpoint every epoch ──
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
            self._save_checkpoint(
                epoch, optimizer, scheduler, output_path,
                global_step, best_loss, 
                wandb_id=wandb.run.id if (use_wandb and wandb.run) else None,
                is_best=is_best, scaler=scaler
            )
            
            # Generate test samples every 5 epochs
            if epoch % 5 == 0 or epoch + 1 == num_epochs:
                self._generate_test_samples(test_dataset, epoch, output_path)
        
        if use_wandb:
            wandb.finish()
        print("Fine-tuning complete!")
        return
    
    def _get_added_cond_kwargs(self, batch_size, h, w, cond):
        """
        SDXL requires additional conditioning kwargs:
        - text_embeds: pooled text embedding [B, 1280]  
        - time_ids: micro-conditioning [B, 6]
        """
        pooled = cond.mean(dim=1)  # [B, context_dim]
        text_embeds = self._pool_proj(pooled)
        text_embeds = torch.clamp(text_embeds, -65000.0, 65000.0).half()
        
        img_size = h * self.vae_scale_factor
        time_ids = torch.tensor(
            [img_size, img_size, 0, 0, img_size, img_size],
            dtype=torch.float16, device=self.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        return {"text_embeds": text_embeds, "time_ids": time_ids}

    def _save_checkpoint(self, epoch, optimizer, scheduler, output_path,
                         global_step=0, best_loss=float('inf'),
                         wandb_id=None, is_best=False, scaler=None):
        ckpt_dir = os.path.join(output_path, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        
        ckpt_data = {
            'epoch': epoch,
            'global_step': global_step,
            'best_loss': best_loss,
            'cond_model': self.cond_model.state_dict(),
            'pool_proj': self._pool_proj.state_dict(),
            'unet_lora': peft.get_peft_model_state_dict(self.unet),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict() if scaler is not None else None,
            'wandb_id': wandb_id,
        }
        
        # Always save latest (overwritten each epoch for resume)
        latest_path = os.path.join(ckpt_dir, 'latest.pth')
        torch.save(ckpt_data, latest_path)
        
        # Save best model separately
        if is_best:
            best_path = os.path.join(ckpt_dir, 'best.pth')
            torch.save(ckpt_data, best_path)
            print(f"Saved checkpoint: epoch {epoch} (best, loss={best_loss:.4f})")
        else:
            print(f"Saved checkpoint: epoch {epoch}")


    @torch.no_grad()
    def generate(self, eeg_dataset, num_samples=5, ddim_steps=None,
                 HW=None, limit=None, state=None, output_path=None):
        """
        Generate images from EEG inputs.
        Compatible with the old eLDM.generate() interface.
        """
        if ddim_steps is None:
            ddim_steps = self.ddim_steps
        
        if output_path:
            os.makedirs(output_path, exist_ok=True)
        
        self.unet.eval()
        self.cond_model.eval()
        
        img_size = 512 // self.vae_scale_factor  # latent size: 64
        shape = (4, img_size, img_size)
        
        all_samples = []
        
        for count, item in enumerate(eeg_dataset):
            if limit is not None and count >= limit:
                break
            
            eeg = item['eeg']
            
            # Handling image shapes:
            # ThingsEEGDataset provides [3, H, W]
            # Old dataset provided [H, W, 3]
            img = item['image']
            if img.shape[0] == 3:
                gt_image = img.unsqueeze(0)  # [1, 3, H, W]
            else:
                gt_image = rearrange(img, 'h w c -> 1 c h w')
            
            print(f"Generating {num_samples} samples in {ddim_steps} steps...")
            
            # Get conditioning from EEG
            eeg_repeated = repeat(eeg, 'h w -> c h w', c=num_samples).to(self.device)
            cond, _ = self.cond_model(eeg_repeated)
            
            # Setup scheduler
            self.scheduler.set_timesteps(ddim_steps)
            
            # Start from random noise
            latents = torch.randn(
                num_samples, *shape,
                device=self.device, dtype=torch.float16
            )
            latents = latents * self.scheduler.init_noise_sigma
            
            # Added conditioning kwargs
            added_cond_kwargs = self._get_added_cond_kwargs(
                num_samples, img_size, img_size, cond
            )
            
            # Denoise
            for t in self.scheduler.timesteps:
                latent_input = self.scheduler.scale_model_input(latents, t)
                noise_pred = self.unet(
                    latent_input,
                    t,
                    encoder_hidden_states=cond.half(),
                    added_cond_kwargs=added_cond_kwargs,
                ).sample
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            # Decode latents to images (FP32 VAE)
            latents_f32 = latents.float() / self.vae.config.scaling_factor
            images = self.vae.decode(latents_f32).sample
            images = torch.clamp((images + 1.0) / 2.0, 0.0, 1.0).float()
            gt_image = torch.clamp((gt_image + 1.0) / 2.0, 0.0, 1.0).float()
            
            all_samples.append(torch.cat([gt_image, images.cpu().float()], dim=0))
            
            # Save individual images
            if output_path:
                samples_np = (255 * torch.cat([gt_image, images.cpu().float()], dim=0).numpy()).astype(np.uint8)
                for idx, img_np in enumerate(samples_np):
                    img_np = rearrange(img_np, 'c h w -> h w c')
                    Image.fromarray(img_np).save(
                        os.path.join(output_path, f'test{count}-{idx}.png')
                    )
        
        # Create grid
        if all_samples:
            grid = torch.stack(all_samples, 0)
            grid = rearrange(grid, 'n b c h w -> (n b) c h w')
            grid = make_grid(grid, nrow=num_samples + 1)
            grid = 255.0 * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            return grid, (255 * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8)
        
        return None, None

    @torch.no_grad()
    def _generate_test_samples(self, test_dataset, epoch, output_path):
        """Generate a few test samples during training for visual inspection."""
        test_dir = os.path.join(output_path, f'samples_epoch{epoch}')
        self.generate(
            test_dataset, num_samples=3, ddim_steps=self.ddim_steps,
            limit=3, output_path=test_dir
        )
        self.unet.train()
        self.cond_model.train()
