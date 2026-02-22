"""
Convert PhysioNet EEG Motor Movement/Imagery .edf files to .npy segments.

Each .edf recording is:
  - Loaded with MNE
  - Filtered to 64 EEG channels (excluding non-EEG channels)
  - Band-pass filtered (0.5 - 45 Hz)
  - Segmented into non-overlapping 512-sample windows
  - Each segment saved as a .npy file  [64, 512]

Usage:
    python preprocess_edf_to_npy.py

Output goes to:  ../dreamdiffusion/datasets/mne_data/
"""

import os
import sys
import mne
import numpy as np
from pathlib import Path
from glob import glob
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──
EDF_ROOT = os.path.join('..', 'dataset', 'files')       # where S001..S109 folders are
OUTPUT_DIR = os.path.join('..', 'dreamdiffusion', 'datasets', 'mne_data')
SEGMENT_LEN = 512       # samples per segment
N_CHANNELS = 64         # target number of EEG channels
LOWCUT = 0.5            # Hz  (high-pass)
HIGHCUT = 45.0          # Hz  (low-pass)


def process_edf(edf_path, output_dir, file_counter):
    """Load one .edf file, segment it, save each segment as .npy."""
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        print(f'  SKIP {edf_path}: {e}')
        return file_counter

    # Pick only EEG channels (PhysioNet has 64 EEG channels)
    eeg_channels = mne.pick_types(raw.info, eeg=True, exclude='bads')
    if len(eeg_channels) < N_CHANNELS:
        print(f'  SKIP {edf_path}: only {len(eeg_channels)} EEG channels (need {N_CHANNELS})')
        return file_counter

    raw.pick(eeg_channels[:N_CHANNELS])

    # Band-pass filter
    raw.filter(LOWCUT, HIGHCUT, verbose=False)

    # Get data: [n_channels, n_samples]
    data = raw.get_data()
    n_channels, n_samples = data.shape

    # Segment into non-overlapping windows of SEGMENT_LEN
    n_segments = n_samples // SEGMENT_LEN
    if n_segments == 0:
        print(f'  SKIP {edf_path}: too short ({n_samples} samples)')
        return file_counter

    for i in range(n_segments):
        segment = data[:, i * SEGMENT_LEN: (i + 1) * SEGMENT_LEN]
        assert segment.shape == (N_CHANNELS, SEGMENT_LEN), f'Bad shape: {segment.shape}'

        # Z-score normalize each segment
        mean = segment.mean()
        std = segment.std()
        if std > 0:
            segment = (segment - mean) / std

        out_path = os.path.join(output_dir, f'seg_{file_counter:06d}.npy')
        np.save(out_path, segment.astype(np.float32))
        file_counter += 1

    return file_counter


def main():
    edf_root = os.path.abspath(EDF_ROOT)
    output_dir = os.path.abspath(OUTPUT_DIR)

    if not os.path.isdir(edf_root):
        print(f'ERROR: EDF root not found: {edf_root}')
        print(f'Expected PhysioNet data at: {edf_root}')
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Find all .edf files
    edf_files = sorted(glob(os.path.join(edf_root, 'S*', '*.edf')))
    print(f'Found {len(edf_files)} .edf files in {edf_root}')

    if len(edf_files) == 0:
        print('No .edf files found!')
        sys.exit(1)

    file_counter = 0
    for idx, edf_path in enumerate(edf_files):
        subject = os.path.basename(os.path.dirname(edf_path))
        fname = os.path.basename(edf_path)
        prev = file_counter
        file_counter = process_edf(edf_path, output_dir, file_counter)
        n_new = file_counter - prev
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f'[{idx+1}/{len(edf_files)}] {subject}/{fname} → {n_new} segments  (total: {file_counter})')

    print(f'\nDone! Saved {file_counter} segments to: {output_dir}')
    print(f'Each segment shape: [{N_CHANNELS}, {SEGMENT_LEN}]')


if __name__ == '__main__':
    main()
