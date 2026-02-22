"""
Stage A1: Pre-train InceptSADEncoder on large-scale EEG data
using temporal masking reconstruction.

Usage:
    python stageA1_incept_pretrain.py
"""

import os
import sys
import copy
import time
import datetime
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config_MBM_EEG
from dataset import eeg_pretrain_dataset
from sc_mbm.incept_encoder import InceptSADEncoder
from sc_mbm.incept_pretrain import InceptSADPretrain


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def fmri_transform(x, sparse_rate=0.2):
    """Data augmentation: randomly zero-out a fraction of time-points."""
    x_aug = copy.deepcopy(x)
    idx = np.random.choice(x.shape[0], int(x.shape[0] * sparse_rate), replace=False)
    x_aug[idx] = 0
    return torch.FloatTensor(x_aug)


def save_model(config, epoch, model, optimizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'checkpoint_epoch{epoch}.pth')
    torch.save({
        'model': model.encoder.state_dict(),         # save only the encoder
        'full_model': model.state_dict(),             # save full model (encoder+decoder)
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'config': config,
    }, save_path)
    print(f'Saved checkpoint: {save_path}')
    return save_path


def train_one_epoch(model, dataloader, optimizer, device, epoch, config):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, data in enumerate(dataloader):
        if isinstance(data, dict):
            eeg = data['eeg'].to(device)
        elif isinstance(data, (list, tuple)):
            eeg = data[0].to(device)
        else:
            eeg = data.to(device)

        # Forward
        loss, pred, mask = model(eeg, mask_ratio=config.mask_ratio)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if config.clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 50 == 0:
            print(f'  Epoch {epoch} [{batch_idx}/{len(dataloader)}] '
                  f'Loss: {loss.item():.6f}')

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    config = Config_MBM_EEG()

    # Override defaults for InceptSAD pre-training
    config.mask_ratio = 0.5          # temporal masking ratio
    config.num_epoch = 200
    config.batch_size = 64
    config.lr = 1e-4
    config.embed_dim = 1024
    config.depth = 6                 # transformer depth
    config.num_heads = 8

    output_path = os.path.join(
        config.root_path, 'results', 'incept_pretrain',
        datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    )
    os.makedirs(output_path, exist_ok=True)
    config.output_path = output_path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # ── dataset ──
    dataset = eeg_pretrain_dataset(
        path='../dreamdiffusion/datasets/mne_data/',
        roi=config.roi,
        patch_size=config.patch_size,
        transform=fmri_transform,
        aug_times=config.aug_times,
        num_sub_limit=config.num_sub_limit,
        include_kam=config.include_kam,
        include_hcp=config.include_hcp
    )
    print(f'Dataset size: {len(dataset)}, Time len: {dataset.data_len}')
    dataloader = DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=True, pin_memory=True, num_workers=4)

    # ── model ──
    encoder = InceptSADEncoder(
        time_len=dataset.data_len,
        in_chans=64,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        global_pool=False,      # need full sequence for reconstruction
    )

    model = InceptSADPretrain(
        encoder=encoder,
        in_chans=64,
        time_len=dataset.data_len,
        decoder_embed_dim=512,
        decoder_depth=4,
        decoder_num_heads=8,
        mask_ratio=config.mask_ratio,
    )
    model = model.to(device)

    # ── optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay
    )

    # ── training loop ──
    print(f'\n{"="*60}')
    print(f'Starting InceptSAD Pre-training')
    print(f'  Device     : {device}')
    print(f'  Epochs     : {config.num_epoch}')
    print(f'  Batch size : {config.batch_size}')
    print(f'  Mask ratio : {config.mask_ratio}')
    print(f'  Embed dim  : {config.embed_dim}')
    print(f'  Depth      : {config.depth}')
    print(f'  Output     : {output_path}')
    print(f'{"="*60}\n')

    start_time = time.time()
    best_loss = float('inf')

    for epoch in range(config.num_epoch):
        avg_loss = train_one_epoch(model, dataloader, optimizer, device, epoch, config)
        print(f'Epoch {epoch}/{config.num_epoch-1}  avg_loss={avg_loss:.6f}')

        # save periodically and at the end
        if epoch % 20 == 0 or epoch + 1 == config.num_epoch:
            last_path = save_model(config, epoch, model, optimizer,
                                   os.path.join(output_path, 'checkpoints'))

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(config, 'best', model, optimizer,
                       os.path.join(output_path, 'checkpoints'))

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f'\nTraining completed in {total_time}')
    print(f'Best loss: {best_loss:.6f}')
    print(f'Final checkpoint: {last_path}')


if __name__ == '__main__':
    main()
