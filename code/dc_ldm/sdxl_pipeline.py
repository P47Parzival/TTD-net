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
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from dc_ldm.ip_adapter_bridge import IPAdapterBridge


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
        self.vae = AutoencoderKL.from_pretrained(
            sdxl_model_id, subfolder="vae", torch_dtype=torch.float16
        ).to(device)
        self.vae.requires_grad_(False)
        self.vae.eval()
        
        print("Loading SDXL UNet...")
        self.unet = UNet2DConditionModel.from_pretrained(
            sdxl_model_id, subfolder="unet", torch_dtype=torch.float16
        ).to(device)
        
        # ── Scheduler: DPM++ 2M Karras ──
        print("Setting up DPM++ 2M Karras scheduler...")
        self.scheduler = DPMSolverMultistepScheduler.from_pretrained(
            sdxl_model_id, subfolder="scheduler",
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
        
        # ── CLIP image encoder for alignment loss ──
        if use_clip_loss:
            print("Loading CLIP image encoder...")
            self.clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                "openai/clip-vit-large-patch14"
            ).to(device)
            self.clip_image_encoder.requires_grad_(False)
            self.clip_image_encoder.eval()
            self.clip_processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
        else:
            self.clip_image_encoder = None
            self.clip_processor = None
        
        # ── EEG Conditioning Module ──
        # SDXL context_dim = 2048 (concatenation of two CLIP encoders)
        self.cond_model = EEGConditioningWrapper(
            eeg_encoder=eeg_encoder,
            pretrained_weights=pretrained_encoder_weights,
            context_dim=self.unet.config.cross_attention_dim,  # 2048 for SDXL
            num_tokens=16,
            clip_dim=768,
            use_clip_loss=use_clip_loss,
        ).to(device)
        
        # VAE scaling factor
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        
        self.use_clip_loss = use_clip_loss
        print(f"EEGtoImageSDXL initialized. UNet cross_attn_dim={self.unet.config.cross_attention_dim}")

    # ── Training ─────────────────────────────
    def finetune(self, dataset, test_dataset, config, output_path):
        """
        Fine-tune the UNet + bridge + EEG encoder on paired EEG-image data.
        """
        os.makedirs(output_path, exist_ok=True)
        
        bs = config.batch_size
        lr = config.lr
        num_epochs = config.num_epoch
        
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, 
                                num_workers=2, pin_memory=True)
        
        # Freeze VAE, train UNet + cond_model
        self.vae.requires_grad_(False)
        self.unet.train()
        self.cond_model.train()
        
        # Optimizer: UNet + conditioning model
        trainable_params = list(self.unet.parameters()) + list(self.cond_model.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
        
        # Mixed precision scaler
        scaler = torch.amp.GradScaler('cuda')
        
        print(f"\n{'='*60}")
        print(f"Starting SDXL Fine-tuning")
        print(f"  Epochs     : {num_epochs}")
        print(f"  Batch size : {bs}")
        print(f"  LR         : {lr}")
        print(f"  CLIP loss  : {self.use_clip_loss}")
        print(f"  Output     : {output_path}")
        print(f"{'='*60}\n")
        
        global_step = 0
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_clip_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(dataloader):
                eeg = batch['eeg'].to(self.device)
                images = batch['image'].to(self.device)  # [B, C, H, W] in [-1, 1]
                
                if images.ndim == 4 and images.shape[1] != 3:
                    images = rearrange(images, 'b h w c -> b c h w')
                
                # Ensure images are float16 for VAE
                images_f16 = images.half()
                
                with torch.no_grad():
                    # Encode images to latent space
                    latents = self.vae.encode(images_f16).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor
                
                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, self.scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=self.device
                ).long()
                
                # Add noise to latents (forward diffusion)
                noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
                
                # Get EEG conditioning
                with torch.amp.autocast('cuda'):
                    cond, raw_eeg_latent = self.cond_model(eeg)
                    
                    # SDXL requires added_cond_kwargs with text_embeds and time_ids
                    # We create dummy ones since we're not using text
                    added_cond_kwargs = self._get_added_cond_kwargs(
                        latents.shape[0], latents.shape[-2], latents.shape[-1],
                        cond
                    )
                    
                    # Predict noise
                    noise_pred = self.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=cond.half(),
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample
                    
                    # Diffusion loss
                    diff_loss = F.mse_loss(noise_pred.float(), noise.float())
                    
                    # CLIP alignment loss
                    clip_loss = torch.tensor(0.0, device=self.device)
                    if self.use_clip_loss and self.clip_image_encoder is not None:
                        with torch.no_grad():
                            # Get CLIP image embeddings
                            clip_input = F.interpolate(
                                (images + 1) / 2,  # [-1,1] → [0,1]
                                size=(224, 224), mode='bilinear', align_corners=False
                            )
                            clip_embeds = self.clip_image_encoder(clip_input.half()).image_embeds
                        clip_loss = self.cond_model.get_clip_loss(raw_eeg_latent, clip_embeds.float())
                    
                    total_loss = diff_loss + 0.5 * clip_loss
                
                # Backward
                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += diff_loss.item()
                epoch_clip_loss += clip_loss.item()
                num_batches += 1
                global_step += 1
                
                if batch_idx % 10 == 0:
                    print(f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                          f"diff_loss={diff_loss.item():.4f} "
                          f"clip_loss={clip_loss.item():.4f}")
            
            avg_loss = epoch_loss / max(num_batches, 1)
            avg_clip = epoch_clip_loss / max(num_batches, 1)
            print(f"Epoch {epoch}/{num_epochs-1}  "
                  f"avg_diff_loss={avg_loss:.4f}  avg_clip_loss={avg_clip:.4f}")
            
            # Save checkpoint
            if epoch % 5 == 0 or epoch + 1 == num_epochs:
                self._save_checkpoint(epoch, optimizer, output_path)
            
            # Generate test samples periodically
            if epoch % 10 == 0 or epoch + 1 == num_epochs:
                self._generate_test_samples(test_dataset, epoch, output_path)
        
        print("Fine-tuning complete!")
        return
    
    def _get_added_cond_kwargs(self, batch_size, h, w, cond):
        """
        SDXL requires additional conditioning kwargs:
        - text_embeds: pooled text embedding [B, 1280]  
        - time_ids: micro-conditioning [B, 6]
        
        Since we use EEG (not text), we derive text_embeds from EEG
        and create default time_ids.
        """
        # Pool the conditioning to create "text_embeds" substitute
        pooled = cond.mean(dim=1)  # [B, context_dim]
        # Project to expected pooled dim (1280 for SDXL)
        if not hasattr(self, '_pool_proj'):
            self._pool_proj = nn.Linear(
                cond.shape[-1], 1280
            ).half().to(self.device)
        text_embeds = self._pool_proj(pooled.half())
        
        # Default time_ids: [original_h, original_w, crop_top, crop_left, target_h, target_w]
        img_size = h * self.vae_scale_factor
        time_ids = torch.tensor(
            [img_size, img_size, 0, 0, img_size, img_size],
            dtype=torch.float16, device=self.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        return {"text_embeds": text_embeds, "time_ids": time_ids}

    def _save_checkpoint(self, epoch, optimizer, output_path):
        ckpt_dir = os.path.join(output_path, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f'checkpoint_epoch{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'cond_model': self.cond_model.state_dict(),
            'unet': self.unet.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, path)
        print(f"Saved checkpoint: {path}")

    # ── Generation ───────────────────────────
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
            gt_image = rearrange(item['image'], 'h w c -> 1 c h w')
            
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
            
            # Decode latents to images
            latents = latents / self.vae.config.scaling_factor
            images = self.vae.decode(latents).sample
            images = torch.clamp((images + 1.0) / 2.0, 0.0, 1.0)
            gt_image = torch.clamp((gt_image + 1.0) / 2.0, 0.0, 1.0)
            
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
