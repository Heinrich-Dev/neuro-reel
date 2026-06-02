"""
Exercise the schema and pepana loader against the real ac1_u004_000.mat
file, and cross-check that the binned output matches what the original
v1_encoder.py pipeline produces.
"""
import os
import sys
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)               # finds the loaders/ package
sys.path.insert(0, os.path.join(here, 'lib'))   # finds v1_encoder.py
MAT = os.path.join(here, 'ac1_u004_000.mat')

from loaders.pepana import load_pepana
from loaders.base import Session


print("=" * 70)
print("1) Load via the new schema")
print("=" * 70)
session = load_pepana(MAT, frames_root='/no/frames/here')   # no stimuli on disk
print(session)
print()
print(f"  unit_ids:    {session.unit_ids}")
print(f"  source:      {session.source}")
print(f"  frame rate:  {session.frame_rate_hz} Hz")
print(f"  frame shape: {session.frame_shape}  (None since no frames available)")

print()
print("=" * 70)
print("2) Schema validation")
print("=" * 70)
issues = session.validate(strict=False)
print(f"  issues found: {issues if issues else 'none'}")

print()
print("=" * 70)
print("3) Trial 0 inspection")
print("=" * 70)
t0 = session.trials[0]
print(f"  trial_id:       {t0.trial_id}")
print(f"  stimulus_key:   {t0.stimulus_key}")
print(f"  duration:       {t0.duration_s:.3f} s")
print(f"  n_units:        {t0.n_units}")
print(f"  spike trains:   {[len(st) for st in t0.spike_times[:4]]}... (per-unit counts)")

# Bin at 30 Hz, the model's native rate
counts = t0.bin_spikes(bin_rate_hz=30.0)
print(f"  binned @ 30Hz:  shape={counts.shape}, total spikes={int(counts.sum())}")

print()
print("=" * 70)
print("4) Cross-check: same bins via the original v1_encoder pipeline")
print("=" * 70)
from v1_encoder import load_session, bin_spikes, FRAME_DURATION

orig_conds = load_session(MAT)
orig0 = orig_conds[0]
print(f"  original movie_id, segment_id: ({orig0['movie_id']}, {orig0['segment_id']})")
print(f"  schema   stimulus_key:         {t0.stimulus_key}")
assert t0.stimulus_key == (orig0['movie_id'], orig0['segment_id'])
print("  ✓ stimulus keys agree")

# Bin both pipelines with the same number of bins and same bin duration,
# and compare. Use the schema's bin count (derived from duration_s).
n_bins = counts.shape[0]
orig_binned = np.stack([
    bin_spikes(st, n_bins, FRAME_DURATION)
    for st in orig0['spike_times']
], axis=1)
max_diff = int(np.abs(orig_binned - counts).max())
n_disagree = int((orig_binned != counts).sum())
print(f"  schema  spike-count matrix: shape={counts.shape}")
print(f"  v1_encoder spike-count matrix: shape={orig_binned.shape}")
print(f"  max per-bin difference: {max_diff}")
print(f"  number of disagreeing bins: {n_disagree} / {counts.size}")

# We expect 1.0 / 30.0 vs 3 / 90.0 to be bit-identical but let's confirm.
print(f"  (bin durations: schema=1/30s={1/30:.10f}, "
      f"v1_encoder=3/90={FRAME_DURATION:.10f})")
if max_diff == 0:
    print("  ✓ binned spike counts are identical")
else:
    print("  ✗ DISAGREEMENT")

print()
print("=" * 70)
print("5) Unique stimulus count and per-stimulus repetition")
print("=" * 70)
uniq = session.unique_stimuli()
print(f"  distinct stimuli in this session: {len(uniq)}")
print(f"  total trials: {len(session.trials)}")
print(f"  repeats per stimulus: {len(session.trials) // len(uniq)}")
print(f"  example unique stimulus_keys: {uniq[:5]}")
