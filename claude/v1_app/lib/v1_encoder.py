"""
V1 Neural Response Prediction from Natural Movie Clips
======================================================
Trains a deep network to predict V1 multi-unit firing rates from the
stimulus frames that drove them, on the Ringach lab macaque dataset
(pepANA .mat files + movie_frames/movieNNN_MMM.images/ JPEG directories).

Architecture: shared CNN core + per-neuron factorized readout
              (Klindt et al. 2017 / Sinz et al. style).
Loss:         Poisson NLL on spike counts per 33.33 ms frame bin.
Evaluation:   Pearson correlation per neuron on held-out segments.
"""

import os
import glob
import numpy as np
from PIL import Image
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

FRAME_DURATION = 3 / 90.0          # 33.33 ms per stimulus frame
LATENCY_SEC    = 0.06              # V1 response lag — tune in {30..100} ms
N_HISTORY      = 8                 # ~266 ms of stimulus history per sample
FRAMES_ROOT    = "movie_frames"

H_IMG, W_IMG   = 240, 320


# ----------------------------------------------------------------------
# 1. Loading neural data from a pepANA .mat file
# ----------------------------------------------------------------------

def load_session(mat_path):
    """Read one pepANA .mat session.

    Returns a list of per-condition dicts:
      {movie_id, segment_id, spike_times: list[np.ndarray]}
    Uses only the first repeat (noRepeats is usually 1 in this dataset).
    """
    mat = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    pep = mat["pepANA"]
    out = []
    for cond in np.atleast_1d(pep.listOfResults):
        symbols = list(np.atleast_1d(cond.symbols))
        values  = list(np.atleast_1d(cond.values))
        params  = {s: int(np.atleast_1d(v).flat[0])
                   for s, v in zip(symbols, values)}
        rep = np.atleast_1d(cond.repeat)[0]
        spike_times = []
        for ch in np.atleast_1d(rep.data):
            # ch is a 1x2 cell: [arrival_times (1xN double), waveforms (48xN int8)]
            times = np.atleast_1d(ch[0]).astype(np.float64).ravel()
            spike_times.append(times)
        out.append({
            "movie_id":    params.get("movie_id", -1),
            "segment_id":  params.get("segment_id", -1),
            "spike_times": spike_times,
        })
    return out


# ----------------------------------------------------------------------
# 2. Loading stimulus frames for one (movie, segment)
# ----------------------------------------------------------------------

def load_segment_frames(movie_id, segment_id, root=FRAMES_ROOT):
    """Return (T, H, W) grayscale float32 array in [0, 1]."""
    d = os.path.join(root, f"movie{movie_id:03d}_{segment_id:03d}.images")
    files = sorted(glob.glob(os.path.join(d, "*.jpeg")))
    frames = np.stack([np.asarray(Image.open(f).convert("L")) for f in files])
    return frames


def bin_spikes(spike_times, n_bins, bin_dur):
    edges = np.arange(n_bins + 1) * bin_dur
    counts, _ = np.histogram(spike_times, bins=edges)
    return counts.astype(np.float32)


# ----------------------------------------------------------------------
# 3. Dataset: aligned (stimulus history, spike-count vector) pairs
# ----------------------------------------------------------------------

class V1Dataset(Dataset):
    """
    Each sample:
      X: (N_HISTORY, H, W)   stimulus history ending LATENCY_SEC before bin k
      y: (n_neurons,)        spike counts in frame-aligned bin k
    """
    def __init__(self, conditions, frames_by_segment,
                 history=N_HISTORY, latency=LATENCY_SEC,
                 crop=None):
        self.frames = frames_by_segment
        self.history = history
        self.bin_dur = FRAME_DURATION
        self.crop = crop
        self.samples = []          # (key, stim_start, stim_end, y_vec)

        lat_bins = int(round(latency / self.bin_dur))
        for cond in conditions:
            key = (cond["movie_id"], cond["segment_id"])
            if key not in frames_by_segment:
                continue
            n_frames = len(frames_by_segment[key])
            y_all = np.stack([
                bin_spikes(st, n_frames, self.bin_dur)
                for st in cond["spike_times"]
            ], axis=1)              # (n_frames, n_neurons)
            for k in range(history + lat_bins, n_frames):
                stim_end   = k - lat_bins
                stim_start = stim_end - history
                self.samples.append((key, stim_start, stim_end, y_all[k]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key, s, e, y = self.samples[idx]
        X = self.frames[key][s:e].astype(np.float32) / 255.0
        if self.crop is not None:
            r0, r1, c0, c1 = self.crop
            X = X[:, r0:r1, c0:c1]
        # z-score each window — simple per-sample normalization
        X = (X - X.mean()) / (X.std() + 1e-6)
        return torch.from_numpy(X.copy()), torch.from_numpy(y)


# ----------------------------------------------------------------------
# 4. Model: CNN core + factorized readout
# ----------------------------------------------------------------------

class Core(nn.Module):
    """Shared CNN: treats N_HISTORY frames as input channels."""
    def __init__(self, in_channels=N_HISTORY, channels=(16, 32, 64)):
        super().__init__()
        c1, c2, c3 = channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c1, 9, padding=4),
            nn.BatchNorm2d(c1), nn.ELU(),
            nn.Conv2d(c1, c2, 7, padding=3),
            nn.BatchNorm2d(c2), nn.ELU(),
            nn.Conv2d(c2, c3, 7, padding=3),
            nn.BatchNorm2d(c3), nn.ELU(),
        )
        self.out_channels = c3

    def forward(self, x):
        return self.net(x)


class FactorizedReadout(nn.Module):
    """Per-neuron readout = (spatial mask H x W) tensor-product (feature vec C)."""
    def __init__(self, n_neurons, channels, h, w):
        super().__init__()
        self.spatial  = nn.Parameter(torch.randn(n_neurons, h, w) * 0.01)
        self.features = nn.Parameter(torch.randn(n_neurons, channels) * 0.01)
        self.bias     = nn.Parameter(torch.zeros(n_neurons))

    def forward(self, feat_map):
        # feat_map: (B, C, H, W) -> (B, N)
        pooled = torch.einsum("bchw,nhw->bnc", feat_map, self.spatial)
        return (pooled * self.features).sum(-1) + self.bias


class V1Predictor(nn.Module):
    def __init__(self, n_neurons, h=H_IMG, w=W_IMG):
        super().__init__()
        self.core = Core()
        self.readout = FactorizedReadout(n_neurons, self.core.out_channels, h, w)

    def forward(self, x):
        return F.softplus(self.readout(self.core(x)))


# ----------------------------------------------------------------------
# 5. Training
# ----------------------------------------------------------------------

def poisson_nll(rate, target, eps=1e-8):
    return (rate - target * torch.log(rate + eps)).mean()


def per_neuron_correlation(pred, target):
    """Pearson r per neuron; NaN-safe for silent neurons."""
    cc = []
    for n in range(pred.shape[1]):
        p, t = pred[:, n], target[:, n]
        if p.std() < 1e-8 or t.std() < 1e-8:
            cc.append(np.nan)
        else:
            cc.append(np.corrcoef(p, t)[0, 1])
    return np.array(cc)


def train(mat_path, n_epochs=50, batch_size=64, lr=3e-4,
          weight_decay=1e-2, l1_spatial=1e-3, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)

    # ---- load
    conditions = load_session(mat_path)
    keys = sorted({(c["movie_id"], c["segment_id"]) for c in conditions})
    print(f"Loaded {len(conditions)} conditions, {len(keys)} unique segments")
    frames_by_segment = {k: load_segment_frames(*k) for k in keys}
    n_neurons = len(conditions[0]["spike_times"])

    # ---- split by segment to prevent temporal leakage
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(keys))
    n_test = max(1, len(keys) // 6)
    n_val  = max(1, len(keys) // 6)
    test_keys  = {keys[i] for i in perm[:n_test]}
    val_keys   = {keys[i] for i in perm[n_test:n_test + n_val]}
    train_keys = set(keys) - test_keys - val_keys

    def split(conds, ks):
        return [c for c in conds if (c["movie_id"], c["segment_id"]) in ks]

    train_ds = V1Dataset(split(conditions, train_keys), frames_by_segment)
    val_ds   = V1Dataset(split(conditions, val_keys),   frames_by_segment)
    test_ds  = V1Dataset(split(conditions, test_keys),  frames_by_segment)
    print(f"train/val/test samples: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, num_workers=0)

    # ---- model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = V1Predictor(n_neurons).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = np.inf
    for epoch in range(n_epochs):
        model.train()
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            rate = model(X)
            loss = poisson_nll(rate, y) + l1_spatial * model.readout.spatial.abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()

        # val
        model.eval()
        val_loss = 0.0; n = 0
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(device), y.to(device)
                rate = model(X)
                val_loss += poisson_nll(rate, y).item() * X.size(0)
                n += X.size(0)
        val_loss /= max(n, 1)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_v1_model.pt")

        print(f"epoch {epoch:02d}  train_loss {loss.item():.4f}  "
              f"val_loss {val_loss:.4f}  best {best_val:.4f}")

    # ---- test
    model.load_state_dict(torch.load("best_v1_model.pt"))
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y in test_dl:
            preds.append(model(X.to(device)).cpu().numpy())
            targets.append(y.numpy())
    preds = np.concatenate(preds); targets = np.concatenate(targets)
    cc = per_neuron_correlation(preds, targets)
    print(f"\nTEST  median r = {np.nanmedian(cc):.3f}   "
          f"mean r = {np.nanmean(cc):.3f}")
    print("Per-neuron r:", np.round(cc, 3))
    return model, cc


if __name__ == "__main__":
    import sys
    mat = sys.argv[1] if len(sys.argv) > 1 else "ac1_u005_007.mat"
    train(mat)
