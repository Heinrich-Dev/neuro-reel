"""
v1_compare.py — Compare predicted firing rates against actual recorded spikes
                for one (movie_id, segment_id) condition.

Usage:
    python3 v1_compare.py predictions.npz neurodata/ac1/ac1_u004_000.mat 0 5

Args:
    predictions.npz   output of v1_predict.py
    .mat file         the session whose checkpoint produced those predictions
    movie_id          which movie (0–3) the predictions correspond to
    segment_id        which segment (0–29) of that movie

Outputs:
    Per-electrode Pearson correlation (predicted vs actual spike counts).
    compare.png — overlaid time courses for 4 active electrodes.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from v1_encoder import load_session, bin_spikes, FRAME_DURATION


def main(pred_path, mat_path, movie_id, segment_id):
    # ---- load predictions
    data = np.load(pred_path)
    preds   = data["predictions"]           # (n_bins, n_neurons)
    times_s = data["times_s"]
    bin_idx = (times_s / FRAME_DURATION).round().astype(int)

    # ---- load the matching condition from the .mat file
    conds = load_session(mat_path)
    match = [c for c in conds
             if c["movie_id"] == movie_id and c["segment_id"] == segment_id]
    if not match:
        sys.exit(f"No condition with movie={movie_id}, segment={segment_id}")
    cond = match[0]
    n_neurons = len(cond["spike_times"])
    if n_neurons != preds.shape[1]:
        sys.exit(f"Mismatch: predictions have {preds.shape[1]} neurons, "
                 f".mat has {n_neurons}")

    # ---- bin actual spikes with the same alignment used in training
    n_bins_full = int(bin_idx.max()) + 1
    actual_full = np.stack([
        bin_spikes(st, n_bins_full, FRAME_DURATION)
        for st in cond["spike_times"]
    ], axis=1)
    actual = actual_full[bin_idx]            # (n_bins_predicted, n_neurons)

    # ---- per-electrode correlation
    print(f"\nmovie={movie_id}, segment={segment_id}, "
          f"{len(bin_idx)} predicted bins\n")
    print(f"{'elec':>4} {'r':>8} {'pred (Hz)':>11} {'actual (Hz)':>13}")
    print("-" * 40)
    rs = []
    for n in range(n_neurons):
        p, a = preds[:, n], actual[:, n]
        if p.std() < 1e-8 or a.std() < 1e-8:
            r = np.nan
        else:
            r = np.corrcoef(p, a)[0, 1]
        rs.append(r)
        print(f"{n:>4} {r:>+8.3f} {p.mean()/FRAME_DURATION:>10.1f}  "
              f"{a.mean()/FRAME_DURATION:>12.1f}")
    print("-" * 40)
    print(f"median r = {np.nanmedian(rs):+.3f}   "
          f"mean r = {np.nanmean(rs):+.3f}")

    # ---- plot 4 most-active electrodes, smoothed for readability
    active = sorted(range(n_neurons), key=lambda n: -actual[:, n].sum())[:4]
    fig, axes = plt.subplots(len(active), 1, figsize=(12, 8), sharex=True)
    for ax, n in zip(axes, active):
        ax.plot(times_s,
                gaussian_filter1d(actual[:, n], 3) / FRAME_DURATION,
                label="actual", alpha=0.8)
        ax.plot(times_s,
                gaussian_filter1d(preds[:, n], 3) / FRAME_DURATION,
                label="predicted", alpha=0.8)
        ax.set_ylabel(f"E{n}\n(Hz)")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    plt.suptitle(f"Predicted vs actual — movie {movie_id}, segment {segment_id}")
    plt.tight_layout()
    plt.savefig("compare.png", dpi=120)
    print(f"\nSaved plot → compare.png")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
