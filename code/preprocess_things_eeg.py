"""
Preprocess ThingsEEG (ds003825) dataset for fine-tuning.

Reads raw BrainVision EEG (.vhdr) + events.tsv, extracts epochs
around each stimulus onset, and saves paired {eeg, image_path, label}
as a single .pth file per subject.

Usage:
    python preprocess_things_eeg.py \
        --things_eeg_dir ../dreamdiffusion/datasets/things_eeg \
        --image_dir ../dreamdiffusion/datasets/object_images \
        --output_dir ../dreamdiffusion/datasets/things_eeg_processed \
        --subjects 1 2 3 4 5

Each subject produces ~22,248 paired (EEG epoch, image path) samples.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import mne
from scipy.interpolate import interp1d


def load_events_tsv(events_path):
    """Load events.tsv and return relevant columns."""
    df = pd.read_csv(events_path, sep='\t')
    print(f"  Loaded {len(df)} events from {os.path.basename(events_path)}")
    return df


def extract_epochs(raw, events_df, epoch_tmin=-0.1, epoch_tmax=0.9,
                   target_len=512, image_dir=None):
    """
    Extract epochs from raw EEG aligned to stimulus onsets.
    
    Args:
        raw: MNE Raw object
        events_df: DataFrame from events.tsv
        epoch_tmin: Start of epoch relative to stimulus onset (seconds)
        epoch_tmax: End of epoch relative to stimulus onset (seconds) 
        target_len: Resample each epoch to this many time points
        image_dir: Root directory of THINGS images
    
    Returns:
        list of dicts: [{eeg, label, object, image_path}, ...]
    """
    sfreq = raw.info['sfreq']
    data = raw.get_data()  # [n_channels, n_samples]
    n_channels, n_samples = data.shape
    
    # Calculate epoch window in samples
    pre_samples = int(abs(epoch_tmin) * sfreq)
    post_samples = int(epoch_tmax * sfreq)
    epoch_len = pre_samples + post_samples
    
    print(f"  EEG: {n_channels} channels, {n_samples} samples, {sfreq} Hz")
    print(f"  Epoch window: {epoch_tmin}s to {epoch_tmax}s = {epoch_len} samples")
    print(f"  Resampling epochs to {target_len} time points")
    
    dataset = []
    skipped = 0
    missing_images = 0
    
    for idx, row in events_df.iterrows():
        onset = int(row['onset'])
        
        # Calculate epoch boundaries
        start = onset - pre_samples
        end = onset + post_samples
        
        # Skip if epoch extends beyond data boundaries
        if start < 0 or end > n_samples:
            skipped += 1
            continue
        
        # Extract epoch: [n_channels, epoch_len]
        epoch = data[:, start:end].copy()
        
        # Pick only EEG channels (first 63 or 64)
        # ThingsEEG uses 63 EEG channels + reference
        if epoch.shape[0] > 64:
            epoch = epoch[:64, :]
        
        # Z-score normalize per channel
        mean = epoch.mean(axis=1, keepdims=True)
        std = epoch.std(axis=1, keepdims=True)
        std[std < 1e-8] = 1e-8
        epoch = (epoch - mean) / std
        
        # Resample to target_len using interpolation
        if epoch.shape[1] != target_len:
            x_old = np.linspace(0, 1, epoch.shape[1])
            x_new = np.linspace(0, 1, target_len)
            f = interp1d(x_old, epoch, axis=1)
            epoch = f(x_new)
        
        # Build image path from 'stim' column
        # stim format: stimuli\carousel\carousel_11s.jpg
        stim_path = str(row['stim']).replace('stimuli\\', '').replace('stimuli/', '')
        # stim_path now: carousel/carousel_11s.jpg or carousel\carousel_11s.jpg
        stim_path = stim_path.replace('\\', '/')
        
        # Full path: image_dir/carousel/carousel_11s.jpg
        if image_dir:
            full_image_path = os.path.join(image_dir, stim_path)
            if not os.path.exists(full_image_path):
                missing_images += 1
                if missing_images <= 5:
                    print(f"  WARNING: Image not found: {full_image_path}")
                continue
        else:
            full_image_path = stim_path
        
        dataset.append({
            'eeg': torch.from_numpy(epoch).float(),    # [n_channels, target_len]
            'label': int(row['objectnumber']),           # 0-1853
            'object': str(row['object']),                # concept name
            'image_path': full_image_path,               # full image path
            'image_name': str(row['stimname']),           # e.g., carousel_11s.jpg
            'subject': 0,                                 # will be set per-subject
        })
    
    print(f"  Extracted {len(dataset)} epochs, skipped {skipped} (boundary), "
          f"missing {missing_images} images")
    return dataset


def process_subject(sub_id, things_eeg_dir, image_dir, output_dir,
                    target_len=512, epoch_tmin=-0.1, epoch_tmax=0.9):
    """
    Process a single subject's EEG data.
    
    Args:
        sub_id: Subject number (1-50)
        things_eeg_dir: Root directory of ThingsEEG dataset
        image_dir: Root directory of THINGS images
        output_dir: Where to save processed .pth files
        target_len: Number of time points per epoch
    """
    sub_str = f"sub-{sub_id:02d}"
    print(f"\n{'='*60}")
    print(f"Processing {sub_str}")
    print(f"{'='*60}")
    
    # Locate files
    eeg_dir = os.path.join(things_eeg_dir, sub_str, 'eeg')
    vhdr_path = os.path.join(eeg_dir, f'{sub_str}_task-rsvp_eeg.vhdr')
    events_path = os.path.join(eeg_dir, f'{sub_str}_task-rsvp_events.tsv')
    
    if not os.path.exists(vhdr_path):
        print(f"  ERROR: {vhdr_path} not found. Skipping.")
        return None
    if not os.path.exists(events_path):
        print(f"  ERROR: {events_path} not found. Skipping.")
        return None
    
    # Load raw EEG with MNE
    print(f"  Loading raw EEG from {os.path.basename(vhdr_path)}...")
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose='ERROR')
    
    # Band-pass filter: 0.5 - 45 Hz (like PhysioNet preprocessing)
    print("  Applying band-pass filter (0.5-45 Hz)...")
    raw.filter(l_freq=0.5, h_freq=45.0, verbose='ERROR')
    
    # Pick only EEG channels (exclude EOG, misc, etc.)
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    raw.pick(eeg_picks)
    print(f"  Selected {len(eeg_picks)} EEG channels")
    
    # Load events
    events_df = load_events_tsv(events_path)
    
    # Extract epochs
    dataset = extract_epochs(
        raw, events_df,
        epoch_tmin=epoch_tmin,
        epoch_tmax=epoch_tmax,
        target_len=target_len,
        image_dir=image_dir,
    )
    
    # Set subject ID
    for item in dataset:
        item['subject'] = sub_id
    
    # Collect unique labels and concept names
    labels = sorted(set(item['label'] for item in dataset))
    objects = sorted(set(item['object'] for item in dataset))
    
    # Verify EEG shape
    if dataset:
        sample = dataset[0]['eeg']
        print(f"\n  Sample EEG shape: {sample.shape}")
        print(f"  Unique concepts: {len(objects)}")
        print(f"  Total epochs: {len(dataset)}")
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{sub_str}_things_eeg.pth')
    
    save_data = {
        'dataset': dataset,
        'labels': labels,
        'objects': objects,
        'subject': sub_id,
        'n_channels': sample.shape[0] if dataset else 0,
        'time_len': target_len,
        'epoch_tmin': epoch_tmin,
        'epoch_tmax': epoch_tmax,
    }
    
    torch.save(save_data, output_path)
    print(f"  Saved to {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1e6:.1f} MB")
    
    return save_data


def main():
    parser = argparse.ArgumentParser(description='Preprocess ThingsEEG for fine-tuning')
    parser.add_argument('--things_eeg_dir', type=str,
                        default='../dreamdiffusion/datasets/things_eeg',
                        help='Path to ThingsEEG BIDS dataset root')
    parser.add_argument('--image_dir', type=str,
                        default='../dreamdiffusion/datasets/object_images',
                        help='Path to THINGS images root')
    parser.add_argument('--output_dir', type=str,
                        default='../dreamdiffusion/datasets/things_eeg_processed',
                        help='Output directory for processed .pth files')
    parser.add_argument('--subjects', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                        help='Subject IDs to process (default: 1 2 3 4 5)')
    parser.add_argument('--target_len', type=int, default=512,
                        help='Number of time points per epoch (default: 512)')
    parser.add_argument('--epoch_tmin', type=float, default=-0.1,
                        help='Epoch start relative to stimulus (seconds)')
    parser.add_argument('--epoch_tmax', type=float, default=0.9,
                        help='Epoch end relative to stimulus (seconds)')
    
    args = parser.parse_args()
    
    print("ThingsEEG Preprocessing")
    print(f"  EEG dir    : {args.things_eeg_dir}")
    print(f"  Image dir  : {args.image_dir}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Subjects   : {args.subjects}")
    print(f"  Target len : {args.target_len}")
    print(f"  Epoch      : [{args.epoch_tmin}s, {args.epoch_tmax}s]")
    
    for sub_id in args.subjects:
        process_subject(
            sub_id=sub_id,
            things_eeg_dir=args.things_eeg_dir,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            target_len=args.target_len,
            epoch_tmin=args.epoch_tmin,
            epoch_tmax=args.epoch_tmax,
        )
    
    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)


if __name__ == '__main__':
    main()