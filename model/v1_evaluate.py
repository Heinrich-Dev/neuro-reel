"""
v1_evaluate.py — Evaluate trained V1 encoder on every held-out test segment.

Computes the same train/val/test split as v1_encoder.py (seed=0), runs
prediction on each of the ~20 test segments, compares against the
recorded spikes, and aggregates per-electrode Pearson correlations.

Usage:
    python3 v1_evaluate.py best_v1_model.pt neurodata/ac1/ac1_u004_000.mat

Optional third arg overrides the frames root (default: movie_frames):
    python3 v1_evaluate.py best_v1_model.pt session.mat path/to/movie_frames

Outputs:
    evaluation.npz   — per-segment, per-electrode correlations + summary
    evaluation.png   — per-electrode boxplot and pooled histogram
    plus a printed summary table
"""

import sys
import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from v1_encoder import (
    V1Predictor, load_session, bin_spikes,
    FRAME_DURATION, N_HISTORY, LATENCY_SEC,
)


def get_test_keys(n_movies=4, n_segments=30, seed=0):
    """Reproduce the test-set split that v1_encoder.train() uses."""
    keys = sorted([(m, s) for m in range(n_movies) for s in range(n_segments)])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(keys))
    n_test = max(1, len(keys) // 6)
    return sorted([keys[i] for i in perm[:n_test]])


def load_frames(movie_id, segment_id, root):
    d = os.path.join(root, f"movie{movie_id:03d}_{segment_id:03d}.images")
    files = sorted(glob.glob(os.path.join(d, "*.jpeg")))
    if not files:
        files = sorted(glob.glob(os.path.join(d, "*.jpg")))
    if not files:
        return None
    frames = np.stack([np.asarray(Image.open(f).convert("L")) for f in files])
    return frames.astype(np.float32) / 255.0


def predict_segment(model, frames, device, batch_size=32):
    """Run the model over one stimulus segment with the same windowing as training."""
    n_frames = len(frames)
    lat_bins = int(round(LATENCY_SEC / FRAME_DURATION))
    windows, bin_idx = [], []
    for k in range(N_HISTORY + lat_bins, n_frames):
        s, e = k - lat_bins - N_HISTORY, k - lat_bins
        w = frames[s:e]
        w = (w - w.mean()) / (w.std() + 1e-6)
        windows.append(w)
        bin_idx.append(k)
    X_all = np.stack(windows).astype(np.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(X_all), batch_size):
            batch = torch.from_numpy(X_all[i:i + batch_size]).to(device)
            out.append(model(batch).cpu().numpy())
    return np.concatenate(out), np.array(bin_idx)


def per_neuron_corr(pred, actual):
    rs = []
    for n in range(pred.shape[1]):
        p, a = pred[:, n], actual[:, n]
        if p.std() < 1e-8 or a.std() < 1e-8:
            rs.append(np.nan)
        else:
            rs.append(np.corrcoef(p, a)[0, 1])
    return np.array(rs)


def evaluate(checkpoint_path, mat_path, frames_root="movie_frames"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- load model (auto-detect shape from checkpoint)
    state = torch.load(checkpoint_path, map_location=device)
    n_neurons, H, W = state["readout.spatial"].shape
    model = V1Predictor(n_neurons=n_neurons, h=H, w=W).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model: {n_neurons} electrodes, {H}x{W} frame readout\n")

    # ---- index recorded conditions by (movie, segment)
    conds = load_session(mat_path)
    cond_by_key = {(c["movie_id"], c["segment_id"]): c for c in conds}

    # ---- iterate over the test-set segments
    test_keys = get_test_keys()
    print(f"Evaluating {len(test_keys)} test segments\n")
    print(f"{'segment':<22} {'median r':>10} {'mean r':>10}  {'bins'}")
    print("-" * 55)

    all_corrs, segment_labels = [], []
    for movie_id, segment_id in test_keys:
        label = f"movie{movie_id:03d}_{segment_id:03d}"
        frames = load_frames(movie_id, segment_id, frames_root)
        if frames is None:
            print(f"{label:<22} {'-':>10} {'-':>10}  (no frames found)")
            continue
        if frames.shape[1:] != (H, W):
            print(f"{label:<22} {'-':>10} {'-':>10}  (size mismatch)")
            continue
        cond = cond_by_key.get((movie_id, segment_id))
        if cond is None:
            print(f"{label:<22} {'-':>10} {'-':>10}  (not in .mat)")
            continue

        preds, bin_idx = predict_segment(model, frames, device)

        n_full = int(bin_idx.max()) + 1
        actual_full = np.stack([
            bin_spikes(st, n_full, FRAME_DURATION) for st in cond["spike_times"]
        ], axis=1)
        actual = actual_full[bin_idx]

        rs = per_neuron_corr(preds, actual)
        all_corrs.append(rs)
        segment_labels.append((movie_id, segment_id))
        print(f"{label:<22} {np.nanmedian(rs):>+10.3f} "
              f"{np.nanmean(rs):>+10.3f}  {len(bin_idx)}")

    if not all_corrs:
        sys.exit("\nNo segments evaluated — check frames_root and .mat path.")

    all_corrs = np.array(all_corrs)        # (n_segments, n_neurons)
    flat = all_corrs[~np.isnan(all_corrs)]

    # ---- summary tables
    print()
    print("=" * 55)
    print(f"SUMMARY  ({len(all_corrs)} segments × {n_neurons} electrodes)")
    print("=" * 55)
    print(f"\n{'electrode':>10} {'median r':>10} {'mean r':>10}")
    print("-" * 35)
    for n in range(n_neurons):
        print(f"{n:>10} {np.nanmedian(all_corrs[:, n]):>+10.3f} "
              f"{np.nanmean(all_corrs[:, n]):>+10.3f}")

    print(f"\nPooled across {len(flat)} (segment, electrode) pairs:")
    print(f"  median r = {np.median(flat):+.3f}")
    print(f"  mean r   = {np.mean(flat):+.3f}")
    print(f"  IQR      = [{np.percentile(flat, 25):+.3f}, "
          f"{np.percentile(flat, 75):+.3f}]")
    print(f"  fraction with r > 0.2 : {(flat > 0.2).mean():.2f}")
    print(f"  fraction with r > 0.4 : {(flat > 0.4).mean():.2f}")

    # ---- save artifacts
    np.savez("evaluation.npz",
             segments=np.array(segment_labels),
             correlations=all_corrs,
             per_elec_median=np.nanmedian(all_corrs, axis=0),
             per_elec_mean=np.nanmean(all_corrs, axis=0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.boxplot([all_corrs[:, n][~np.isnan(all_corrs[:, n])]
                 for n in range(n_neurons)],
                positions=range(n_neurons), showmeans=True)
    ax1.axhline(0, color='k', linestyle=':', alpha=0.4)
    ax1.set_xlabel("electrode index")
    ax1.set_ylabel("Pearson r (over test segments)")
    ax1.set_title(f"Per-electrode test accuracy ({len(all_corrs)} segments)")

    ax2.hist(flat, bins=25, edgecolor='k', alpha=0.7)
    ax2.axvline(np.median(flat), color='r', linestyle='--',
                label=f"median = {np.median(flat):+.3f}")
    ax2.axvline(0, color='k', linestyle=':')
    ax2.set_xlabel("Pearson r")
    ax2.set_ylabel("count")
    ax2.set_title("Pooled (segment × electrode) correlations")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("evaluation.png", dpi=120)
    print("\nSaved → evaluation.npz, evaluation.png")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    frames_root = sys.argv[3] if len(sys.argv) > 3 else "movie_frames"
    evaluate(sys.argv[1], sys.argv[2], frames_root)
