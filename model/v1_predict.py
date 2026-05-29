"""
v1_predict.py — Run a trained V1 encoder on a movie segment.

Usage:
    python3 v1_predict.py best_v1_model.pt movie_frames/movie000_005.images

Outputs:
    predictions.npz  — saved predicted firing rates and timing info
    prints a preview of the prediction matrix
"""

import sys
import os
import glob
import numpy as np
import torch
from PIL import Image

# Reuse classes and constants from the training script
from v1_encoder import (
    V1Predictor,
    FRAME_DURATION,    # 33.33 ms per bin
    N_HISTORY,         # 8-frame stimulus window
    LATENCY_SEC,       # 60 ms response delay
)


def load_frames(image_dir):
    """Load all .jpeg frames from a movieNNN_MMM.images directory."""
    files = sorted(glob.glob(os.path.join(image_dir, "*.jpeg")))
    if not files:
        # fall back to .jpg if needed
        files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not files:
        raise FileNotFoundError(f"No frames found in {image_dir}")
    frames = np.stack([np.asarray(Image.open(f).convert("L")) for f in files])
    return frames.astype(np.float32) / 255.0   # (T, H, W) in [0, 1]


def predict(checkpoint_path, image_dir, output_path="predictions.npz"):
    # ---- load frames
    frames = load_frames(image_dir)
    n_frames, H, W = frames.shape
    print(f"Loaded {n_frames} frames of size {H}x{W} from {image_dir}")

    # ---- load model, auto-detecting n_neurons / spatial size from checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(checkpoint_path, map_location=device)
    n_neurons, h_ckpt, w_ckpt = state["readout.spatial"].shape
    print(f"Checkpoint: {n_neurons} electrodes, readout for {h_ckpt}x{w_ckpt} frames")

    if (h_ckpt, w_ckpt) != (H, W):
        raise ValueError(
            f"Frame size {H}x{W} does not match what the model was trained on "
            f"({h_ckpt}x{w_ckpt}). Resize frames or use matching stimuli."
        )

    model = V1Predictor(n_neurons=n_neurons, h=H, w=W).to(device)
    model.load_state_dict(state)
    model.eval()

    # ---- build sliding-window inputs
    # For each bin k we want to predict, the stimulus is frames
    # [k - lat - history : k - lat], matching how training was done.
    lat_bins = int(round(LATENCY_SEC / FRAME_DURATION))
    windows, bin_indices = [], []
    for k in range(N_HISTORY + lat_bins, n_frames):
        s, e = k - lat_bins - N_HISTORY, k - lat_bins
        w = frames[s:e]
        w = (w - w.mean()) / (w.std() + 1e-6)   # same per-window z-score as training
        windows.append(w)
        bin_indices.append(k)

    X_all = np.stack(windows).astype(np.float32)
    print(f"Input  shape: {X_all.shape}    # (n_bins, history, H, W)")
    
    # ---- run model in batches to fit in GPU memory
    batch_size = 32
    preds_list = []
    with torch.no_grad():
        for i in range(0, len(X_all), batch_size):
            batch = torch.from_numpy(X_all[i:i+batch_size]).to(device)
            preds_list.append(model(batch).cpu().numpy())
    preds = np.concatenate(preds_list, axis=0)

    print(f"Output shape: {preds.shape}        # (n_bins, n_electrodes)")

    rates_hz = preds / FRAME_DURATION
    times_s  = np.array(bin_indices) * FRAME_DURATION

    print("\nFirst 5 bins x first 5 electrodes (predicted spike counts):")
    print(np.round(preds[:5, :5], 3))
    print(f"\nPer-electrode mean firing rate (Hz): "
          f"{np.round(rates_hz.mean(axis=0), 2)}")

    np.savez(output_path,
             predictions=preds,        # (n_bins, n_electrodes), spike counts per bin
             rates_hz=rates_hz,        # same, converted to Hz
             times_s=times_s,          # bin center times in seconds
             bin_duration_s=FRAME_DURATION)
    print(f"\nSaved → {output_path}")
    return preds, times_s


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    predict(sys.argv[1], sys.argv[2])
