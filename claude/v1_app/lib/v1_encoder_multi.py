"""
V1 Neural Response Prediction — multi-session version
=====================================================

Extends v1_encoder.py to train one shared CNN core across many pepANA .mat
files, each with its own per-session readout head.

Architecture (Sinz / Klindt-style):
    - One Core (CNN). Shared across all sessions.
    - One FactorizedReadout per session, sized to that session's electrode count.

Training:
    Per-step single-session batches. At each step:
      1. Sample a session (proportional to sqrt of its size to keep big
         sessions from dominating without starving small ones).
      2. Draw a batch from that session's loader.
      3. Forward through the shared core + that session's readout.
      4. Backprop. Both the core and the chosen readout receive gradients.

Backward compatibility:
    - The single-session `V1Predictor` class is still defined for inference.
    - `export_session_checkpoint()` extracts one session's readout + the shared
      core into a state dict in the original single-session format, so the
      web app and `v1_predict.py` keep working unchanged.

Usage:
    python3 v1_encoder_multi.py path/to/data_root
    # data_root contains any number of .mat files; each one becomes a session.
"""

import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Reuse helpers from the original encoder so the loading logic stays in one place.
from v1_encoder import (
    FRAME_DURATION, LATENCY_SEC, N_HISTORY, H_IMG, W_IMG, FRAMES_ROOT,
    load_session, load_segment_frames, bin_spikes,
    V1Dataset, Core, FactorizedReadout,
    poisson_nll, per_neuron_correlation,
)


# ----------------------------------------------------------------------
# Multi-session model
# ----------------------------------------------------------------------

class MultiSessionV1Predictor(nn.Module):
    """One shared core, one factorized readout per session.

    Parameters
    ----------
    session_neurons : dict[str, int]
        Maps session_id -> number of recorded electrodes in that session.
    h, w : int
        Frame dimensions the model is built for. All sessions must use the
        same frame size (the Core has fixed convolutional layers).
    """

    def __init__(self, session_neurons, h=H_IMG, w=W_IMG):
        super().__init__()
        self.core = Core()
        self.readouts = nn.ModuleDict({
            sid: FactorizedReadout(n, self.core.out_channels, h, w)
            for sid, n in session_neurons.items()
        })
        self.session_neurons = dict(session_neurons)
        self.h, self.w = h, w

    def forward(self, x, session_id):
        if session_id not in self.readouts:
            raise KeyError(f"Unknown session_id '{session_id}'. "
                           f"Known: {list(self.readouts.keys())}")
        return F.softplus(self.readouts[session_id](self.core(x)))


def export_session_checkpoint(model, session_id, out_path):
    """Extract one session's readout glued to the shared core, save as a
    state dict compatible with the original single-session `V1Predictor`."""
    if session_id not in model.readouts:
        raise KeyError(f"Unknown session_id '{session_id}'")
    sd = {}
    # core.* keys
    for k, v in model.core.state_dict().items():
        sd[f"core.{k}"] = v.cpu().clone()
    # readout.* keys (drop the session prefix)
    for k, v in model.readouts[session_id].state_dict().items():
        sd[f"readout.{k}"] = v.cpu().clone()
    torch.save(sd, out_path)
    return out_path


# ----------------------------------------------------------------------
# Discovering sessions in a data directory
# ----------------------------------------------------------------------

def discover_sessions(data_root, glob_pattern="*.mat"):
    """Return a list of (session_id, mat_path) for every .mat in data_root.

    session_id is just the filename without extension; it gets used as a
    dict key in the ModuleDict, so it must be a valid Python identifier
    fragment. Dots and dashes get replaced with underscores.
    """
    paths = sorted(glob.glob(os.path.join(data_root, "**", glob_pattern),
                              recursive=True))
    out = []
    for p in paths:
        base = os.path.splitext(os.path.basename(p))[0]
        sid = base.replace(".", "_").replace("-", "_")
        out.append((sid, p))
    return out


# ----------------------------------------------------------------------
# Per-session data preparation
# ----------------------------------------------------------------------

def prepare_session(mat_path, frames_root=FRAMES_ROOT, seed=0,
                    val_frac=1 / 6, test_frac=1 / 6, verbose=True):
    """Load one .mat session, split by (movie, segment), build datasets.

    Returns a dict:
        {
          'n_neurons': int,
          'train': V1Dataset, 'val': V1Dataset, 'test': V1Dataset,
        }
    or None if the session can't be built (e.g. no matching frames).
    """
    try:
        conditions = load_session(mat_path)
    except Exception as e:
        if verbose:
            print(f"  ! {os.path.basename(mat_path)}: failed to load ({e})")
        return None

    # Only keep natural-movie conditions (random-seed sessions need different stim handling)
    movie_conds = [c for c in conditions
                   if c.get("movie_id", -1) >= 0 and c.get("segment_id", -1) >= 0]
    if not movie_conds:
        if verbose:
            print(f"  ! {os.path.basename(mat_path)}: no natural-movie conditions")
        return None

    keys = sorted({(c["movie_id"], c["segment_id"]) for c in movie_conds})

    # Load frames; drop segments whose frame folder is missing.
    frames_by_segment = {}
    for k in keys:
        try:
            frames_by_segment[k] = load_segment_frames(*k, root=frames_root)
        except Exception:
            pass
    keys = [k for k in keys if k in frames_by_segment]
    if not keys:
        if verbose:
            print(f"  ! {os.path.basename(mat_path)}: no matching frame folders "
                  f"in {frames_root}")
        return None

    n_neurons = len(movie_conds[0]["spike_times"])

    # Held-out split by segment (no temporal leakage).
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(keys))
    n_test = max(1, int(round(len(keys) * test_frac)))
    n_val  = max(1, int(round(len(keys) * val_frac)))
    test_keys  = {keys[i] for i in perm[:n_test]}
    val_keys   = {keys[i] for i in perm[n_test:n_test + n_val]}
    train_keys = set(keys) - test_keys - val_keys

    def split(conds, ks):
        return [c for c in conds
                if (c["movie_id"], c["segment_id"]) in ks]

    return {
        "n_neurons": n_neurons,
        "train": V1Dataset(split(movie_conds, train_keys), frames_by_segment),
        "val":   V1Dataset(split(movie_conds, val_keys),   frames_by_segment),
        "test":  V1Dataset(split(movie_conds, test_keys),  frames_by_segment),
    }


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train_multi(data_root, n_epochs=50, batch_size=64, lr=3e-4,
                weight_decay=1e-2, l1_spatial=1e-3, seed=0,
                save_path="best_v1_multi.pt",
                export_default_session=None,
                frames_root=FRAMES_ROOT):
    """Train one shared core + per-session readouts on every .mat in data_root.

    export_default_session : optional session_id whose state dict should be
        exported in single-session format after training (for app deployment).
        Defaults to the largest session.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- Discover and prepare sessions ----
    pairs = discover_sessions(data_root)
    if not pairs:
        raise SystemExit(f"No .mat files found under {data_root}")
    print(f"Found {len(pairs)} .mat file(s) under {data_root}")

    sessions = {}     # session_id -> {'n_neurons', 'train', 'val', 'test', 'loaders'}
    for sid, path in pairs:
        info = prepare_session(path, frames_root=frames_root, seed=seed)
        if info is None:
            continue
        sessions[sid] = info
        print(f"  ✓ {sid}: {info['n_neurons']} units  "
              f"train/val/test = {len(info['train'])}/{len(info['val'])}/{len(info['test'])}")
    if not sessions:
        raise SystemExit("No usable sessions found. Check frame folders exist "
                         f"under {frames_root}.")

    # ---- Build model: shared core + one readout per session ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    session_neurons = {sid: s["n_neurons"] for sid, s in sessions.items()}
    model = MultiSessionV1Predictor(session_neurons).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---- Data loaders ----
    for s in sessions.values():
        s["loaders"] = {
            "train": DataLoader(s["train"], batch_size=batch_size, shuffle=True,
                                num_workers=0, drop_last=True),
            "val":   DataLoader(s["val"],   batch_size=batch_size, num_workers=0),
            "test":  DataLoader(s["test"],  batch_size=batch_size, num_workers=0),
        }

    # ---- Sampling weights: sqrt of train-set size keeps big sessions from
    #      dominating while not starving small ones.
    sids = list(sessions.keys())
    weights = np.array([np.sqrt(len(sessions[s]["train"])) for s in sids])
    weights = weights / weights.sum()
    print(f"\nSession sampling weights: "
          f"{ {s: round(float(w), 3) for s, w in zip(sids, weights)} }")

    # Compute steps-per-epoch as the total train batches across all sessions,
    # so an "epoch" still corresponds to roughly one pass through the data.
    steps_per_epoch = sum(len(s["loaders"]["train"]) for s in sessions.values())
    print(f"Steps per epoch: {steps_per_epoch}")

    # ---- Training loop ----
    best_val = np.inf
    rng = np.random.default_rng(seed + 1)
    train_iters = {s: iter(sessions[s]["loaders"]["train"]) for s in sids}

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            sid = rng.choice(sids, p=weights)
            try:
                X, y = next(train_iters[sid])
            except StopIteration:
                train_iters[sid] = iter(sessions[sid]["loaders"]["train"])
                X, y = next(train_iters[sid])

            X, y = X.to(device), y.to(device)
            rate = model(X, sid)
            loss = poisson_nll(rate, y) + \
                   l1_spatial * model.readouts[sid].spatial.abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        # ---- Validation: full pass over every session, weighted equally ----
        model.eval()
        val_losses = {}
        with torch.no_grad():
            for sid in sids:
                total, n = 0.0, 0
                for X, y in sessions[sid]["loaders"]["val"]:
                    X, y = X.to(device), y.to(device)
                    rate = model(X, sid)
                    total += poisson_nll(rate, y).item() * X.size(0)
                    n += X.size(0)
                val_losses[sid] = total / max(n, 1)
        mean_val = float(np.mean(list(val_losses.values())))

        if mean_val < best_val:
            best_val = mean_val
            torch.save({
                "model_state":     model.state_dict(),
                "session_neurons": session_neurons,
                "h": H_IMG, "w": W_IMG,
            }, save_path)

        print(f"epoch {epoch:02d}  train_loss {epoch_loss/steps_per_epoch:.4f}  "
              f"val(mean) {mean_val:.4f}  best {best_val:.4f}  "
              f"per-session val: { {s: round(v, 3) for s, v in val_losses.items()} }")

    # ---- Test ----
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("\n" + "=" * 70)
    print("TEST per session (Pearson r, median ± mean):")
    print("=" * 70)
    summary = {}
    for sid in sids:
        preds, targets = [], []
        with torch.no_grad():
            for X, y in sessions[sid]["loaders"]["test"]:
                preds.append(model(X.to(device), sid).cpu().numpy())
                targets.append(y.numpy())
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        cc = per_neuron_correlation(preds, targets)
        summary[sid] = (float(np.nanmedian(cc)), float(np.nanmean(cc)))
        print(f"  {sid:30s}  median r = {np.nanmedian(cc):+.3f}   "
              f"mean r = {np.nanmean(cc):+.3f}")

    # ---- Export one session in single-session format for app deployment ----
    if export_default_session is None:
        export_default_session = max(sids, key=lambda s: session_neurons[s])
    if export_default_session in sids:
        export_path = save_path.replace(".pt", f".single_{export_default_session}.pt")
        export_session_checkpoint(model, export_default_session, export_path)
        print(f"\nExported single-session checkpoint for '{export_default_session}'")
        print(f"  -> {export_path}")
        print(f"  Drop this in as best_v1_model.pt in the web app to deploy.")

    return model, summary


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Train one shared V1 core across many pepANA sessions.")
    p.add_argument("data_root",
                   help="Directory containing pepANA .mat files (searched recursively)")
    p.add_argument("--frames-root", default=FRAMES_ROOT,
                   help="Directory containing movieNNN_MMM.images/ folders")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default="best_v1_multi.pt")
    p.add_argument("--export", default=None,
                   help="Session ID to export as single-session checkpoint. "
                        "Default: the session with the most electrodes.")
    args = p.parse_args()

    train_multi(
        data_root=args.data_root,
        frames_root=args.frames_root,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        save_path=args.save,
        export_default_session=args.export,
    )


if __name__ == "__main__":
    main()
