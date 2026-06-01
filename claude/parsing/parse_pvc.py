"""
Generic parser for CRCNS pvc-1 style pepANA .mat files.

Walks a directory (default: ./data) and parses every .mat file found,
returning a list of per-file results plus convenience accessors.

Each file is expected to follow the naming convention:
    {prefix}_u{unit:03d}_{block:03d}.mat
e.g. ac1_u004_000.mat, ad1_u012_001.mat

Stimulus type is inferred from the condition variables:
    {'movie_id', 'segment_id'}  -> 'natural_movie'
    {'rseed'}                   -> 'random_noise'
    anything else               -> 'unknown'
"""
import os
import re
import glob
from pathlib import Path

import numpy as np
import scipy.io as sio


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

FNAME_RE = re.compile(r'^(?P<prefix>[a-zA-Z]+\d+)_u(?P<unit>\d+)_(?P<block>\d+)\.mat$')


def parse_filename(fname):
    """Extract (prefix, unit, block) from a pvc-style filename.

    Returns a dict, or None if the filename does not match the convention.
    """
    m = FNAME_RE.match(os.path.basename(fname))
    if not m:
        return None
    return {
        'prefix': m.group('prefix'),
        'unit_id': int(m.group('unit')),
        'block_id': int(m.group('block')),
    }


# ---------------------------------------------------------------------------
# Stimulus-type inference
# ---------------------------------------------------------------------------

STIMULUS_TYPES = {
    frozenset({'movie_id', 'segment_id'}): 'natural_movie',
    frozenset({'rseed'}):                  'random_noise',
}


def infer_stimulus_type(condition_symbols):
    return STIMULUS_TYPES.get(frozenset(condition_symbols), 'unknown')


# ---------------------------------------------------------------------------
# Core loader (single file)
# ---------------------------------------------------------------------------

def load_pepana(path):
    """Load a single pepANA .mat file into a clean Python dict.

    Robust to per-file variations:
      - any set of condition variables (movie_id/segment_id, rseed, etc.)
      - any number of repeats per trial
      - any number of recorded units
      - empty unit slots
    """
    raw = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    pep = raw['pepANA']
    cfg = pep.config

    # ---- metadata ----
    fname_info = parse_filename(path) or {}
    meta = {
        'path':        os.path.abspath(path),
        'filename':    os.path.basename(path),
        'prefix':      fname_info.get('prefix'),
        'unit_id':     fname_info.get('unit_id'),
        'block_id':    fname_info.get('block_id'),
        'animal':      str(cfg.animal)      if hasattr(cfg, 'animal')      else None,
        'unit':        int(cfg.unit)        if hasattr(cfg, 'unit')        else None,
        'experiment':  int(cfg.experiment)  if hasattr(cfg, 'experiment')  else None,
        'hemisphere':  str(cfg.hemisphere)  if hasattr(cfg, 'hemisphere')  else None,
        'penetration': int(cfg.penetration) if hasattr(cfg, 'penetration') else None,
        'electrodes':  np.asarray(pep.elec_list, dtype=int).ravel().tolist(),
        'n_units':     int(np.asarray(pep.elec_list).size),
        'n_trials':    int(pep.no_conditions),
    }

    # ---- per-trial data ----
    trials = []
    list_of_results = np.atleast_1d(pep.listOfResults)
    for i, t in enumerate(list_of_results):
        symbols = [str(s) for s in np.atleast_1d(t.symbols)]
        values = [int(v) if np.ndim(v) == 0 else int(np.asarray(v).ravel()[0])
                  for v in np.atleast_1d(t.values)]
        condition = dict(zip(symbols, values))

        reps = np.atleast_1d(t.repeat)
        repeats = []
        for r in reps:
            units = []
            for u in np.atleast_1d(r.data):
                if not hasattr(u, '__len__') or len(u) == 0:
                    spike_times = np.array([], dtype=float)
                    waveforms = np.zeros((0, 0), dtype=np.int8)
                else:
                    spike_times = np.asarray(u[0]).ravel().astype(float)
                    waveforms = np.asarray(u[1])
                    if waveforms.ndim == 1:
                        waveforms = waveforms.reshape(-1, 1)
                units.append({'spike_times': spike_times,
                              'waveforms':   waveforms})
            repeats.append({
                'tzero_us':  float(r.tzero),
                'sync':      float(r.sync) if np.ndim(r.sync) == 0 else float(np.asarray(r.sync).ravel()[0]),
                'syncCount': int(r.syncCount),
                'units':     units,
            })

        trials.append({'trial_index': i,
                       'condition':   condition,
                       'repeats':     repeats})

    # Infer stimulus type from the first trial's condition variables
    if trials:
        stim_type = infer_stimulus_type(trials[0]['condition'].keys())
    else:
        stim_type = 'unknown'
    meta['stimulus_type'] = stim_type
    meta['condition_variables'] = list(trials[0]['condition'].keys()) if trials else []

    return {'meta': meta, 'trials': trials}


# ---------------------------------------------------------------------------
# Directory walker
# ---------------------------------------------------------------------------

def load_directory(data_dir='data', recursive=True, verbose=True):
    """Parse every .mat file under `data_dir` and return a list of results.

    Each entry is the dict produced by load_pepana(), plus a top-level
    'error' key if parsing failed (in which case the rest of the dict
    contains only the filename info we could recover).

    Parameters
    ----------
    data_dir   : str or Path  -- directory to walk
    recursive  : bool         -- if True, walk subdirectories too
    verbose    : bool         -- print a one-line summary per file
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir.resolve()}")

    pattern = '**/*.mat' if recursive else '*.mat'
    paths = sorted(data_dir.glob(pattern))

    if verbose:
        print(f"Found {len(paths)} .mat file(s) under {data_dir.resolve()}")

    results = []
    for p in paths:
        try:
            parsed = load_pepana(str(p))
            results.append(parsed)
            if verbose:
                m = parsed['meta']
                print(f"  ✓ {m['filename']:30s}  "
                      f"animal={m['animal']:>4} unit={m['unit']}  "
                      f"{m['n_units']:2d} electrodes, "
                      f"{m['n_trials']:3d} trials, "
                      f"stim={m['stimulus_type']}")
        except Exception as e:
            err_entry = {
                'meta': {**(parse_filename(str(p)) or {}),
                         'path': str(p.resolve()),
                         'filename': p.name},
                'trials': [],
                'error':  f'{type(e).__name__}: {e}',
            }
            results.append(err_entry)
            if verbose:
                print(f"  ✗ {p.name:30s}  ERROR: {err_entry['error']}")

    return results


# ---------------------------------------------------------------------------
# Convenience accessors over a parsed collection
# ---------------------------------------------------------------------------

def filter_by(results, **constraints):
    """Return a sub-list of results whose meta matches all kwarg constraints.

    Example:
        movies = filter_by(results, stimulus_type='natural_movie')
        ac1    = filter_by(results, prefix='ac1')
    """
    out = []
    for r in results:
        if 'error' in r:
            continue
        if all(r['meta'].get(k) == v for k, v in constraints.items()):
            out.append(r)
    return out


def summary_table(results):
    """Print a compact table summarizing every successfully parsed file."""
    ok = [r for r in results if 'error' not in r]
    bad = [r for r in results if 'error' in r]

    print(f"\n{'='*92}")
    print(f"{'file':<28} {'prefix':<6} {'unit':>4} {'blk':>4} "
          f"{'electrodes':>10} {'trials':>6}  {'stimulus':<14} {'cond vars'}")
    print(f"{'-'*92}")
    for r in ok:
        m = r['meta']
        print(f"{m['filename']:<28} {str(m['prefix']):<6} "
              f"{m['unit']!s:>4} {m['block_id']!s:>4} "
              f"{m['n_units']:>10} {m['n_trials']:>6}  "
              f"{m['stimulus_type']:<14} "
              f"{', '.join(m['condition_variables'])}")
    print(f"{'='*92}")
    print(f"Parsed OK: {len(ok)}    Failed: {len(bad)}    Total: {len(results)}")
    if bad:
        print("\nFailures:")
        for r in bad:
            print(f"  {r['meta'].get('filename', '?')}: {r.get('error')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data'
    results = load_directory(data_dir, recursive=True, verbose=True)
    summary_table(results)
