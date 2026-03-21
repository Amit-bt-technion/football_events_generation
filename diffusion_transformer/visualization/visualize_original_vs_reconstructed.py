#!/usr/bin/env python3
"""
Visualize original event sequences side-by-side with their autoencoder
reconstructions.  Produces two matched sets of GIFs:

    <output_dir>/original/seq_0000.gif
    <output_dir>/reconstructed/seq_0000.gif

so that seq_0000 in both folders shows the same source sequence.

Usage (minimal – relies on cached data from a prior main.py run):

    python visualize_original_vs_reconstructed.py \
        --num_gifs 10 \
        --autoencoder_path diffusion_transformer/models/autoencoder.pt
"""

import argparse
import os
import sys
import random
from pathlib import Path

import io
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------------------
# Make project-root imports work regardless of working directory
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from extract_event_sequences import build_event_mapping, get_event_name
from diffusion_transformer.data.preprocessing import load_and_embed_matches
from diffusion_transformer.data.event_autoencoder_model import EventAutoencoder

# ---------------------------------------------------------------------------
# Constants (shared with visualizer.py)
# ---------------------------------------------------------------------------
FIELD_IMAGE_PATH = os.path.join(_PROJECT_ROOT, 'diffusion_transformer/visualization/football_field.jpg')
FIELD_BOUNDS = {'left': 24, 'right': 588, 'top': 18, 'bottom': 374}
TEAM_COLORS = {0: '#1f77b4', 1: '#d62728'}


# ---------------------------------------------------------------------------
# GIF creation for a single sequence
# ---------------------------------------------------------------------------
def _render_sequence_frames(field_img, seq_len, pxs, pys, event_names, teams,
                            title=None):
    """
    Render each event as an independent matplotlib figure and return a list
    of PIL Images.  Each frame is drawn from scratch -- no artist reuse.
    """
    frames = []
    for i in range(seq_len):
        fig, ax = plt.subplots(figsize=(10, 6.7))
        ax.imshow(field_img)
        ax.axis('off')

        color = TEAM_COLORS[teams[i]]
        ax.scatter([pxs[i]], [pys[i]], s=200, zorder=5,
                   facecolors=color, edgecolors='white', linewidths=1.5)
        ax.text(pxs[i], pys[i] - 8, event_names[i], fontsize=10,
                fontweight='bold', ha='center', va='bottom', color='white',
                bbox=dict(boxstyle='round,pad=0.3', alpha=0.85,
                          facecolor=color, edgecolor='none'))
        ax.text(0.02, 0.97, f'Event {i + 1}/{seq_len}',
                transform=ax.transAxes, fontsize=9, va='top', color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))
        if title:
            ax.text(0.98, 0.97, title, transform=ax.transAxes,
                    fontsize=11, fontweight='bold', va='top', ha='right',
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#333333', alpha=0.7))

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=80)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
    return frames


def create_sequence_gif(sequence, boundaries, field_img, out_path, title=None):
    """
    Render a single 128-d event sequence as an animated GIF on the field photo.

    Args:
        sequence:   np.ndarray (seq_len, 128)
        boundaries: event-type decoding boundaries from build_event_mapping()
        field_img:  pre-loaded field image array
        out_path:   destination .gif path
        title:      optional text shown in the top-right corner
    """
    seq_len = len(sequence)
    frame_duration_ms = int(60_000 / seq_len)

    event_names, pxs, pys, teams = [], [], [], []
    for vec in sequence:
        name, _, _ = get_event_name(vec[0], boundaries)
        event_names.append(name)
        x_m, y_m = vec[2] * 120, vec[3] * 80
        pxs.append(FIELD_BOUNDS['left'] + (x_m / 120) * (FIELD_BOUNDS['right'] - FIELD_BOUNDS['left']))
        pys.append(FIELD_BOUNDS['top'] + (y_m / 80) * (FIELD_BOUNDS['bottom'] - FIELD_BOUNDS['top']))
        teams.append(0 if vec[12] < 0.75 else 1)

    frames = _render_sequence_frames(field_img, seq_len, pxs, pys,
                                     event_names, teams, title=title)
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=frame_duration_ms, loop=0)


# ---------------------------------------------------------------------------
# Sequence sampling
# ---------------------------------------------------------------------------
def sample_sequences(events_dict, num_sequences, seq_len, seed=42):
    """
    Sample random contiguous windows of *seq_len* events from the pool of
    all matches, spread across different matches.

    Returns:
        sequences: np.ndarray of shape (num_sequences, seq_len, 128)
        metadata:  list of (match_id, start_event_index) tuples
    """
    rng = np.random.RandomState(seed)

    # Build pool of all valid (match_id, start_index) pairs
    candidates = []
    for mid, events in events_dict.items():
        n = len(events)
        if n < seq_len:
            continue
        for s in range(n - seq_len + 1):
            candidates.append((mid, s))

    chosen = rng.choice(len(candidates), size=min(num_sequences, len(candidates)), replace=False)

    sequences = []
    metadata = []
    for idx in chosen:
        mid, s = candidates[idx]
        sequences.append(events_dict[mid][s:s + seq_len])
        metadata.append((mid, int(s)))

    return np.stack(sequences), metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize original vs autoencoder-reconstructed event sequences"
    )
    p.add_argument("--num_gifs", type=int, default=10)
    p.add_argument("--sequence_length", type=int, default=50)
    p.add_argument("--output_dir", type=str, default="outputs/original_vs_reconstructed")
    p.add_argument("--autoencoder_path", type=str,
                   default="diffusion_transformer/models/autoencoder.pt")
    p.add_argument("--csv_dir", type=str, default="csv")
    p.add_argument("--cache_dir", type=str, default="cache")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force_recompute", action="store_true", default=False)
    p.add_argument("--verbose", action="store_true", default=False)
    p.add_argument("--task", type=str, default="full")
    p.add_argument("--task_params", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    # ---- load data ---------------------------------------------------------
    print("Loading match events (uses cache if available)...")
    events_dict, _ = load_and_embed_matches(args)
    print(f"Loaded {len(events_dict)} matches")

    # ---- sample sequences --------------------------------------------------
    print(f"Sampling {args.num_gifs} sequences of length {args.sequence_length}...")
    original, seq_metadata = sample_sequences(events_dict, args.num_gifs,
                                              args.sequence_length, seed=args.seed)
    print(f"Sampled {len(original)} sequences  shape={original.shape}")
    for i, (mid, start) in enumerate(seq_metadata):
        print(f"  seq {i}: match_id={mid}  start_event={start}")

    # ---- autoencoder reconstruct -------------------------------------------
    print(f"Loading autoencoder from {args.autoencoder_path}...")
    ae = EventAutoencoder(input_dim=original.shape[-1], latent_dim=32)
    ae.load_state_dict(torch.load(args.autoencoder_path, map_location=device,
                                  weights_only=False))
    ae = ae.to(device).eval()

    flat = torch.tensor(original.reshape(-1, original.shape[-1]),
                        dtype=torch.float32).to(device)
    with torch.no_grad():
        reconstructed_flat = ae.decoder(ae.encoder(flat)).cpu().numpy()
    reconstructed = reconstructed_flat.reshape(original.shape)
    print("Reconstruction complete")

    # ---- prepare dirs & shared resources -----------------------------------
    orig_dir = os.path.join(args.output_dir, 'original')
    recon_dir = os.path.join(args.output_dir, 'reconstructed')
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)

    if not os.path.exists(FIELD_IMAGE_PATH):
        print(f"ERROR: field image not found at {FIELD_IMAGE_PATH}")
        return

    field_img = plt.imread(FIELD_IMAGE_PATH)
    _, _, _, boundaries = build_event_mapping()

    # ---- create GIFs -------------------------------------------------------
    for i in range(len(original)):
        match_id, start_event = seq_metadata[i]
        tag = f"{match_id}_{start_event}"
        print(f"  [{i+1}/{len(original)}] {tag}")

        create_sequence_gif(original[i], boundaries, field_img,
                            os.path.join(orig_dir, f'{tag}.gif'),
                            title='Original')
        create_sequence_gif(reconstructed[i], boundaries, field_img,
                            os.path.join(recon_dir, f'{tag}.gif'),
                            title='Reconstructed')

    print(f"\nDone! GIFs saved to:\n  {orig_dir}\n  {recon_dir}")


if __name__ == '__main__':
    main()
