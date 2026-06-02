"""
V1 Neural Response Prediction — multi-session, schema-driven version
====================================================================

Trains one shared CNN core across many recording sessions, with a per-session
factorized readout head. Sessions are consumed as `loaders.base.Session`
objects, so any dataset whose loader produces a Session — pvc-1 pepANA,
Allen Neuropixels, NWB, etc. — works without changes here.

Backward compatible with the original single-session inference pipeline:
the exported single-session checkpoint loads cleanly into the existing
`V1Predictor` class used by the web app's inference.py.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Make sibling packages importable when run directly from the app root.
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "lib"), os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from v1_encoder import (
    FRAME_DURATION, LATENCY_SEC, N_HISTORY, H_IMG, W_IMG,
    Core, FactorizedReadout,
    poisson_nll, per_neuron_correlation,
)
from loaders.base import Session, Trial
from loaders.pepana import load_pepana


# ----------------------------------------------------------------------
# Schema-aware dataset
# ----------------------------------------------------------------------

class SessionDataset(Dataset):
    """Build (stimulus_history, spike_count) training pairs from a Session.

    For each trial we generate sliding-window samples aligned the same way
    the original V1Dataset did: the stimulus window covers N_HISTORY frames
    ending `latency` seconds before the bin we're predicting.

    Parameters
    ----------
    session : Session
        Source recording. Stimulus frames are pulled via session.frames_provider.
    history : int
        How many frames of stimulus history feed into each prediction.
    latency : float
        Response delay in seconds between stimulus and spike.
    preload_frames : bool
        If True, load all unique stimuli once at construction time and cache
        them in memory. Faster training and avoids LRU thrashing, but uses
        ~frames * H * W bytes. Set False for huge stimulus corpora (Allen).
    target_hw : (H, W) or None
        Resize all frames to this size at load time. Required if you'll be
        mixing sessions with different native resolutions.
    """

    def __init__(self, session, history=N_HISTORY, latency=LATENCY_SEC,
                 preload_frames=True, target_hw=None):
        self.session = session
        self.history = history
        self.bin_dur = 1.0 / session.frame_rate_hz
        self.target_hw = target_hw

        if target_hw is not None:
            from PIL import Image
            def resize(frames):
                H, W = target_hw
                out = np.empty((frames.shape[0], H, W), dtype=frames.dtype)
                for i in range(frames.shape[0]):
                    im = Image.fromarray(frames[i]).resize((W, H), Image.BILINEAR)
                    out[i] = np.asarray(im)
                return out
            self._maybe_resize = resize
        else:
            self._maybe_resize = lambda x: x

        # Optionally preload every distinct stimulus once
        self.frame_cache = {}
        if preload_frames:
            for key in session.unique_stimuli():
                try:
                    frames = session.frames_provider(key)
                except FileNotFoundError:
                    continue   # skip stimuli without disk frames
                self.frame_cache[key] = self._maybe_resize(frames)

        # Build (trial_idx, stim_start, stim_end, spike_count_vec) sample index
        lat_bins = int(round(latency / self.bin_dur))
        self.samples = []
        for ti, trial in enumerate(session.trials):
            frames = self._get_frames(trial.stimulus_key)
            if frames is None:
                continue
            n_frames = len(frames)
            counts = trial.bin_spikes(bin_rate_hz=session.frame_rate_hz,
                                      n_bins=n_frames)
            for k in range(history + lat_bins, n_frames):
                self.samples.append((ti, k - lat_bins - history, k - lat_bins,
                                     counts[k]))

    def _get_frames(self, stimulus_key):
        """Look up frames from cache or fall back to the provider."""
        if stimulus_key in self.frame_cache:
            return self.frame_cache[stimulus_key]
        try:
            frames = self.session.frames_provider(stimulus_key)
            return self._maybe_resize(frames)
        except FileNotFoundError:
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ti, s, e, y = self.samples[idx]
        frames = self._get_frames(self.session.trials[ti].stimulus_key)
        X = frames[s:e].astype(np.float32) / 255.0
        X = (X - X.mean()) / (X.std() + 1e-6)
        return torch.from_numpy(X.copy()), torch.from_numpy(y.copy())


# ----------------------------------------------------------------------
# Multi-session model
# ----------------------------------------------------------------------

class MultiSessionV1Predictor(nn.Module):
    """One shared core, one factorized readout per session."""

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
    """Extract one session's readout + the shared core into the single-session
    state-dict format `V1Predictor` expects (for web-app deployment)."""
    if session_id not in model.readouts:
        raise KeyError(f"Unknown session_id '{session_id}'")
    sd = {}
    for k, v in model.core.state_dict().items():
        sd[f"core.{k}"] = v.cpu().clone()
    for k, v in model.readouts[session_id].state_dict().items():
        sd[f"readout.{k}"] = v.cpu().clone()
    torch.save(sd, out_path)
    return out_path


# ----------------------------------------------------------------------
# Source discovery and session prep
# ----------------------------------------------------------------------

def discover_pepana(data_root):
    """Walk a directory for .mat files. Returns list of paths."""
    import glob
    return sorted(glob.glob(os.path.join(data_root, "**", "*.mat"),
                            recursive=True))


def prepare_session(session, seed=0, val_frac=1/6, test_frac=1/6,
                    target_hw=None, preload_frames=True, verbose=True):
    """Build train/val/test SessionDataset splits from one Session.

    Held-out split is by stimulus_key, not by trial — prevents temporal
    leakage when the same stimulus is shown multiple times.
    """
    keys = session.unique_stimuli()
    if not keys:
        if verbose:
            print(f"  ! {session.session_id}: no stimuli")
        return None

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(keys))
    n_test = max(1, int(round(len(keys) * test_frac)))
    n_val  = max(1, int(round(len(keys) * val_frac)))
    test_keys  = {keys[i] for i in perm[:n_test]}
    val_keys   = {keys[i] for i in perm[n_test:n_test + n_val]}
    train_keys = set(keys) - test_keys - val_keys

    def subset(ks):
        sub = Session(
            session_id      = session.session_id,
            source          = session.source,
            stimulus_type   = session.stimulus_type,
            n_units         = session.n_units,
            unit_ids        = session.unit_ids,
            unit_metadata   = session.unit_metadata,
            frame_rate_hz   = session.frame_rate_hz,
            frame_shape     = session.frame_shape,
            frames_provider = session.frames_provider,
            trials          = [t for t in session.trials if t.stimulus_key in ks],
            metadata        = session.metadata,
        )
        return SessionDataset(sub, preload_frames=preload_frames,
                              target_hw=target_hw)

    out = {
        "n_neurons": session.n_units,
        "train": subset(train_keys),
        "val":   subset(val_keys),
        "test":  subset(test_keys),
    }
    if verbose:
        print(f"  ✓ {session.session_id}: {session.n_units} units  "
              f"train/val/test = {len(out['train'])}/{len(out['val'])}/{len(out['test'])}")
    return out


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train_multi(sessions_data, n_epochs=50, batch_size=64, lr=3e-4,
                weight_decay=1e-2, l1_spatial=1e-3, seed=0,
                save_path="best_v1_multi.pt",
                export_default_session=None):
    """Train shared core + per-session readouts.

    Parameters
    ----------
    sessions_data : dict[str, dict]
        session_id -> {'n_neurons', 'train', 'val', 'test'}, as produced by
        prepare_session().
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if not sessions_data:
        raise SystemExit("No usable sessions provided.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    session_neurons = {sid: s["n_neurons"] for sid, s in sessions_data.items()}
    model = MultiSessionV1Predictor(session_neurons).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Per-session data loaders
    for s in sessions_data.values():
        s["loaders"] = {
            "train": DataLoader(s["train"], batch_size=batch_size, shuffle=True,
                                num_workers=0, drop_last=True),
            "val":   DataLoader(s["val"],   batch_size=batch_size, num_workers=0),
            "test":  DataLoader(s["test"],  batch_size=batch_size, num_workers=0),
        }

    # Sampling: sqrt of train size keeps big sessions from dominating
    sids = list(sessions_data.keys())
    weights = np.array([np.sqrt(len(sessions_data[s]["train"])) for s in sids])
    weights = weights / weights.sum()
    print(f"Session sampling weights: "
          f"{ {s: round(float(w), 3) for s, w in zip(sids, weights)} }")

    steps_per_epoch = sum(len(s["loaders"]["train"]) for s in sessions_data.values())
    print(f"Steps per epoch: {steps_per_epoch}")

    # Training loop
    best_val = np.inf
    rng = np.random.default_rng(seed + 1)
    train_iters = {s: iter(sessions_data[s]["loaders"]["train"]) for s in sids}

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            sid = rng.choice(sids, p=weights)
            try:
                X, y = next(train_iters[sid])
            except StopIteration:
                train_iters[sid] = iter(sessions_data[sid]["loaders"]["train"])
                X, y = next(train_iters[sid])

            X, y = X.to(device), y.to(device)
            rate = model(X, sid)
            loss = (poisson_nll(rate, y)
                    + l1_spatial * model.readouts[sid].spatial.abs().mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        # Validation: full pass over every session, weighted equally
        model.eval()
        val_losses = {}
        with torch.no_grad():
            for sid in sids:
                total, n = 0.0, 0
                for X, y in sessions_data[sid]["loaders"]["val"]:
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

    # Test
    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("\n" + "=" * 70)
    print("TEST per session (Pearson r):")
    print("=" * 70)
    summary = {}
    for sid in sids:
        preds, targets = [], []
        with torch.no_grad():
            for X, y in sessions_data[sid]["loaders"]["test"]:
                preds.append(model(X.to(device), sid).cpu().numpy())
                targets.append(y.numpy())
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        cc = per_neuron_correlation(preds, targets)
        summary[sid] = (float(np.nanmedian(cc)), float(np.nanmean(cc)))
        print(f"  {sid:30s}  median r = {np.nanmedian(cc):+.3f}   "
              f"mean r = {np.nanmean(cc):+.3f}")

    # Export one session in single-session format for web-app deployment
    if export_default_session is None:
        export_default_session = max(sids, key=lambda s: session_neurons[s])
    if export_default_session in sids:
        export_path = save_path.replace(".pt", f".single_{export_default_session}.pt")
        export_session_checkpoint(model, export_default_session, export_path)
        print(f"\nExported single-session checkpoint for '{export_default_session}'")
        print(f"  -> {export_path}")

    return model, summary


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Train one shared V1 core across many sessions.")
    p.add_argument("data_root", help="Directory of .mat files (searched recursively)")
    p.add_argument("--frames-root", default="movie_frames",
                   help="Directory of movieNNN_MMM.images/ folders")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default="best_v1_multi.pt")
    p.add_argument("--export", default=None,
                   help="Session ID to export as single-session checkpoint. "
                        "Default: the session with the most electrodes.")
    p.add_argument("--no-preload-frames", action="store_true",
                   help="Disable preloading stimulus frames (use for big corpora)")
    args = p.parse_args()

    paths = discover_pepana(args.data_root)
    if not paths:
        raise SystemExit(f"No .mat files found under {args.data_root}")
    print(f"Found {len(paths)} .mat file(s) under {args.data_root}")

    sessions_data = {}
    for path in paths:
        try:
            session = load_pepana(path, frames_root=args.frames_root)
        except Exception as e:
            print(f"  ! {os.path.basename(path)}: failed to load ({e})")
            continue
        prepped = prepare_session(session, seed=args.seed,
                                  preload_frames=not args.no_preload_frames)
        if prepped is not None:
            sessions_data[session.session_id] = prepped

    train_multi(
        sessions_data    = sessions_data,
        n_epochs         = args.epochs,
        batch_size       = args.batch_size,
        lr               = args.lr,
        seed             = args.seed,
        save_path        = args.save,
        export_default_session = args.export,
    )


if __name__ == "__main__":
    main()
