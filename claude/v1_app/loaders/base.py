"""
Intermediate representation for V1 response-prediction datasets.

Every loader (pvc-1 pepANA, Allen Neuropixels, NWB, ...) produces a `Session`
conforming to this schema. The training code consumes `Session` objects and
never sees a native file format directly. To add a new dataset, write a
loader function that returns `Session` — no changes to model or training code.

Design notes
------------
* `Trial.spike_times` is raw (in seconds, relative to trial onset), not
  pre-binned. The bin alignment is a downstream decision that depends on
  what frame rate the model is being trained at, and rebinning is cheap.
  Allen and pepANA can both convert their natively-different time bases
  into this format.

* `frames_provider` is a callable, not a pre-loaded array. Large stimulus
  corpora (Allen `natural_movie_three` is ~3600 frames per session) would
  otherwise dominate memory. Loaders are expected to cache internally.

* `stimulus_key` is whatever hashable tuple uniquely identifies the
  stimulus shown on a trial. pepANA uses `(movie_id, segment_id)`; Allen
  natural movies might use `(movie_name, start_frame, end_frame)`. The
  schema doesn't constrain the shape — just that two trials showing the
  same stimulus produce the same key.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any
import numpy as np


@dataclass
class Trial:
    """One stimulus presentation paired with its recorded spike trains."""
    trial_id:     str
    stimulus_key: tuple                 # hashable identifier for the stimulus
    spike_times:  list                  # list[np.ndarray]; one array of times (s) per unit
    duration_s:   float
    metadata:     dict = field(default_factory=dict)

    @property
    def n_units(self) -> int:
        return len(self.spike_times)

    def bin_spikes(self, bin_rate_hz: float, n_bins: int = None) -> np.ndarray:
        """Bin this trial's spike trains. Returns (n_bins, n_units) float32."""
        bin_dur = 1.0 / bin_rate_hz
        if n_bins is None:
            n_bins = int(round(self.duration_s / bin_dur))
        edges = np.arange(n_bins + 1) * bin_dur
        out = np.empty((n_bins, self.n_units), dtype=np.float32)
        for i, st in enumerate(self.spike_times):
            out[:, i], _ = np.histogram(st, bins=edges)
        return out


@dataclass
class Session:
    """One recording session with its trials, units, and stimulus access."""
    session_id:      str
    source:          str                    # e.g. "pvc-1-pepana", "allen-neuropixels"
    stimulus_type:   str                    # "natural_movie", "random_noise", ...
    n_units:         int
    unit_ids:        list                   # list[str]; per-unit identifiers
    unit_metadata:   list                   # list[dict]; electrode, area, QC, ...
    frame_rate_hz:   float                  # native stimulus presentation rate
    frame_shape:     tuple                  # native (H, W); (None, None) if unknown
    frames_provider: Callable               # stimulus_key -> np.ndarray (T, H, W)
    trials:          list                   # list[Trial]
    metadata:        dict = field(default_factory=dict)

    def validate(self, strict: bool = True) -> list:
        """Check internal consistency. Returns list of issues; raises if strict."""
        issues = []

        if len(self.unit_ids) != self.n_units:
            issues.append(
                f"unit_ids has {len(self.unit_ids)} entries, n_units={self.n_units}")
        if len(self.unit_metadata) != self.n_units:
            issues.append(
                f"unit_metadata has {len(self.unit_metadata)} entries, n_units={self.n_units}")

        for i, t in enumerate(self.trials):
            if t.n_units != self.n_units:
                issues.append(
                    f"trial {i}: {t.n_units} unit spike trains, expected {self.n_units}")
            if t.duration_s <= 0:
                issues.append(f"trial {i}: non-positive duration {t.duration_s}")
            if not isinstance(t.stimulus_key, tuple):
                issues.append(f"trial {i}: stimulus_key must be tuple, got {type(t.stimulus_key)}")

        if self.frame_rate_hz <= 0:
            issues.append(f"frame_rate_hz must be positive, got {self.frame_rate_hz}")

        if strict and issues:
            raise ValueError("Session validation failed:\n  " + "\n  ".join(issues))
        return issues

    def unique_stimuli(self) -> list:
        """Distinct stimulus_keys across all trials, in stable order."""
        seen, out = set(), []
        for t in self.trials:
            if t.stimulus_key not in seen:
                seen.add(t.stimulus_key)
                out.append(t.stimulus_key)
        return out

    def __repr__(self):
        return (f"Session('{self.session_id}', source='{self.source}', "
                f"{self.n_units} units, {len(self.trials)} trials, "
                f"stimulus='{self.stimulus_type}', {self.frame_rate_hz:g} Hz)")
