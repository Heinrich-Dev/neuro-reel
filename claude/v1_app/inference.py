"""
Inference wrapper around the V1Predictor model.
Adapts the logic in v1_predict.py to be callable from a server context.
"""
import os
import sys
import glob
import numpy as np
import torch
from PIL import Image

# Make lib/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))
from v1_encoder import V1Predictor, FRAME_DURATION, N_HISTORY, LATENCY_SEC

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), 'best_v1_model.pt')

# Cache the loaded model in-process so we don't reload on every request
_MODEL_CACHE = {}


def _get_model(device='cpu'):
    """Load the model once and reuse."""
    if 'model' in _MODEL_CACHE:
        return _MODEL_CACHE['model'], _MODEL_CACHE['n_neurons'], _MODEL_CACHE['hw']

    state = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    n_neurons, h, w = state['readout.spatial'].shape
    model = V1Predictor(n_neurons=n_neurons, h=h, w=w).to(device)
    model.load_state_dict(state)
    model.eval()
    _MODEL_CACHE['model']     = model
    _MODEL_CACHE['n_neurons'] = n_neurons
    _MODEL_CACHE['hw']        = (h, w)
    return model, n_neurons, (h, w)


def load_frames(image_dir):
    """Load all .jpg/.jpeg frames as a grayscale float32 array in [0, 1]."""
    files = sorted(glob.glob(os.path.join(image_dir, '*.jpeg')))
    if not files:
        files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    if not files:
        raise FileNotFoundError(f'No .jpg or .jpeg frames found in {image_dir}')
    frames = np.stack([np.asarray(Image.open(f).convert('L')) for f in files])
    return frames.astype(np.float32) / 255.0


def run_inference(image_dir, device=None, batch_size=16, progress_cb=None):
    """Predict V1 firing rates from a directory of stimulus frames.

    Returns dict with:
      predictions   (n_bins, n_electrodes)  -- predicted spike counts per bin
      rates_hz      (n_bins, n_electrodes)  -- same in Hz
      times_s       (n_bins,)               -- bin centre time within clip
      bin_duration_s float                  -- 1/30 ish
      n_frames_input int                    -- frames actually loaded

    progress_cb: optional fn(fraction_done, message) called during batching.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, n_neurons, (h_ckpt, w_ckpt) = _get_model(device)

    frames = load_frames(image_dir)
    n_frames, H, W = frames.shape
    if (H, W) != (h_ckpt, w_ckpt):
        raise ValueError(
            f'Frame size {H}x{W} does not match the model ({h_ckpt}x{w_ckpt}). '
            'Re-extract frames with --size 320x240.')

    lat_bins = int(round(LATENCY_SEC / FRAME_DURATION))
    first_bin = N_HISTORY + lat_bins
    if first_bin >= n_frames:
        raise ValueError(
            f'Only {n_frames} frames extracted; need at least {first_bin + 1}.')

    bin_indices = list(range(first_bin, n_frames))
    n_bins_out = len(bin_indices)

    # Stream batches: build the windows for each batch on the fly so we never
    # hold all 900 windows in memory at once.
    preds_out = np.empty((n_bins_out, n_neurons), dtype=np.float32)
    n_batches = (n_bins_out + batch_size - 1) // batch_size
    with torch.no_grad():
        for bi, i in enumerate(range(0, n_bins_out, batch_size)):
            ks = bin_indices[i:i + batch_size]
            batch_np = np.empty((len(ks), N_HISTORY, H, W), dtype=np.float32)
            for j, k in enumerate(ks):
                s, e = k - lat_bins - N_HISTORY, k - lat_bins
                w = frames[s:e]
                batch_np[j] = (w - w.mean()) / (w.std() + 1e-6)
            batch = torch.from_numpy(batch_np).to(device)
            preds_out[i:i + len(ks)] = model(batch).cpu().numpy()
            del batch, batch_np
            if progress_cb and (bi % 10 == 0 or bi == n_batches - 1):
                progress_cb((bi + 1) / n_batches,
                            f'Inferring batch {bi + 1}/{n_batches}')
    preds = preds_out

    rates_hz = preds / FRAME_DURATION
    times_s = np.array(bin_indices, dtype=float) * FRAME_DURATION

    return {
        'predictions':    preds,
        'rates_hz':       rates_hz,
        'times_s':        times_s,
        'bin_duration_s': FRAME_DURATION,
        'n_frames_input': int(n_frames),
        'n_electrodes':   int(n_neurons),
    }
