"""
Loader for CRCNS pvc-1 / Ringach-lab pepANA .mat files.

Reads a single .mat file and returns a `Session` conforming to the
loaders.base schema. Frame data is loaded lazily via the frames_provider.
"""
import os
import glob
import numpy as np
from PIL import Image
import scipy.io as sio

from .base import Trial, Session


# pvc-1 movies were played at 30 fps with each frame held for 3 monitor
# refreshes (99.8 Hz / 3 ≈ 33.3 Hz effective rate). The training code in
# v1_encoder.py uses 3/90.0 ≈ 33.33 ms bins, which matches.
DEFAULT_FRAME_RATE_HZ = 90.0 / 3.0      # 30 Hz exactly
DEFAULT_FRAMES_ROOT   = "movie_frames"


def _make_frames_provider(frames_root: str, cache_size: int = 8):
    """Return a callable that loads `(T, H, W)` frames for a stimulus_key.

    Caches the most recently used `cache_size` stimuli in memory so repeated
    accesses (which the training loop will do) don't re-read JPEGs.
    """
    cache = {}
    order = []

    def provider(stimulus_key):
        if stimulus_key in cache:
            return cache[stimulus_key]
        movie_id, segment_id = stimulus_key
        d = os.path.join(frames_root,
                         f"movie{movie_id:03d}_{segment_id:03d}.images")
        files = sorted(glob.glob(os.path.join(d, "*.jpeg")))
        if not files:
            files = sorted(glob.glob(os.path.join(d, "*.jpg")))
        if not files:
            raise FileNotFoundError(f"No frames found at {d}")
        frames = np.stack([np.asarray(Image.open(f).convert("L")) for f in files])
        cache[stimulus_key] = frames
        order.append(stimulus_key)
        if len(order) > cache_size:
            cache.pop(order.pop(0), None)
        return frames

    return provider


def _peek_frame_shape(provider, stimulus_key):
    """Get (H, W) from one stimulus folder; return (None, None) on failure."""
    try:
        f = provider(stimulus_key)
        return (int(f.shape[1]), int(f.shape[2]))
    except FileNotFoundError:
        return (None, None)


def load_pepana(mat_path: str,
                frames_root: str = DEFAULT_FRAMES_ROOT,
                frame_rate_hz: float = DEFAULT_FRAME_RATE_HZ,
                only_natural_movies: bool = True) -> Session:
    """Load a pepANA .mat file into a `Session`.

    Parameters
    ----------
    mat_path : path to the .mat file
    frames_root : directory containing movieNNN_MMM.images/ folders
    frame_rate_hz : presentation rate; default matches pvc-1 (30 Hz)
    only_natural_movies : if True, skip trials whose condition variables
        don't include (movie_id, segment_id). Set False to include rseed
        random-noise trials, but note that those need a separate stimulus
        generator to provide frames; the default frames_provider here only
        knows about movieNNN_MMM.images folders.
    """
    raw = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    pep = raw["pepANA"]
    cfg = pep.config

    session_id = os.path.splitext(os.path.basename(mat_path))[0]
    electrodes = np.asarray(pep.elec_list, dtype=int).ravel().tolist()
    n_units = len(electrodes)

    animal      = str(cfg.animal)      if hasattr(cfg, "animal")      else None
    hemisphere  = str(cfg.hemisphere)  if hasattr(cfg, "hemisphere")  else None
    penetration = int(cfg.penetration) if hasattr(cfg, "penetration") else None
    experiment  = int(cfg.experiment)  if hasattr(cfg, "experiment")  else None

    unit_ids = [f"e{e}" for e in electrodes]
    unit_metadata = [
        {"electrode":   e,
         "animal":      animal,
         "hemisphere":  hemisphere,
         "penetration": penetration,
         "area":        "V1"}
        for e in electrodes
    ]

    trials = []
    observed_type = None

    for i, cond in enumerate(np.atleast_1d(pep.listOfResults)):
        symbols = list(np.atleast_1d(cond.symbols))
        values  = list(np.atleast_1d(cond.values))
        params  = {str(s): int(np.atleast_1d(v).flat[0])
                   for s, v in zip(symbols, values)}

        if "movie_id" in params and "segment_id" in params:
            this_type    = "natural_movie"
            stimulus_key = (params["movie_id"], params["segment_id"])
        elif "rseed" in params:
            this_type    = "random_noise"
            stimulus_key = (params["rseed"],)
        else:
            this_type    = "unknown"
            stimulus_key = tuple(sorted(params.items()))

        if only_natural_movies and this_type != "natural_movie":
            continue
        observed_type = observed_type or this_type

        # Per-unit spike times (this dataset has noRepeats=1 in all sessions we've seen)
        rep = np.atleast_1d(cond.repeat)[0]
        spike_times = []
        for ch in np.atleast_1d(rep.data):
            if hasattr(ch, "__len__") and len(ch) >= 1:
                t = np.atleast_1d(ch[0]).astype(np.float64).ravel()
            else:
                t = np.array([], dtype=np.float64)
            spike_times.append(t)

        # Duration: the .mat file doesn't explicitly store segment duration,
        # so we use the latest spike across all units as a lower bound. This
        # is the same proxy the original parser used. The consumer can pass
        # an explicit n_bins to bin_spikes() if it knows the true duration
        # (e.g., from the frames_provider's frame count).
        latest = [float(t[-1]) for t in spike_times if t.size]
        duration_s = max(latest) if latest else 0.0

        trials.append(Trial(
            trial_id     = f"{session_id}_t{i:03d}",
            stimulus_key = stimulus_key,
            spike_times  = spike_times,
            duration_s   = duration_s,
            metadata     = {"original_index": i, **params},
        ))

    frames_provider = _make_frames_provider(frames_root)
    frame_shape = (None, None)
    if trials:
        frame_shape = _peek_frame_shape(frames_provider, trials[0].stimulus_key)

    return Session(
        session_id      = session_id,
        source          = "pvc-1-pepana",
        stimulus_type   = observed_type or "natural_movie",
        n_units         = n_units,
        unit_ids        = unit_ids,
        unit_metadata   = unit_metadata,
        frame_rate_hz   = frame_rate_hz,
        frame_shape     = frame_shape,
        frames_provider = frames_provider,
        trials          = trials,
        metadata        = {
            "mat_path":   mat_path,
            "animal":     animal,
            "experiment": experiment,
        },
    )
