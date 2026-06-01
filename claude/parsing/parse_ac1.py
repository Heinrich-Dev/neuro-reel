import scipy.io as sio
import numpy as np
 
def load_pepana(path):
    """Load a pepANA .mat file and return a clean Python dict."""
    raw = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    pep = raw['pepANA']
 
    # --- metadata / experiment config ---
    cfg = pep.config
    meta = {
        'animal':      str(cfg.animal),
        'unit':        int(cfg.unit),
        'experiment':  int(cfg.experiment),
        'hemisphere':  str(cfg.hemisphere),
        'penetration': int(cfg.penetration),
        'refresh_hz':  float(cfg.display[3]) if hasattr(cfg.display, '__len__') else None,
        'electrodes':  np.asarray(pep.elec_list, dtype=int).ravel().tolist(),
        'n_units':     int(np.asarray(pep.elec_list).size),
        'n_trials':    int(pep.no_conditions),
    }
 
    # --- per-trial info ---
    # listOfResults is length n_trials; each entry has condition symbols/values + repeat(s).
    # Each repeat contains .data, a length-N_units array of (spike_times, waveforms) pairs.
    trials = []
    for i, t in enumerate(pep.listOfResults):
        # Condition labels (e.g. movie_id, segment_id) and their values
        symbols = [str(s) for s in np.atleast_1d(t.symbols)]
        values  = [int(v) if np.ndim(v) == 0 else int(np.asarray(v).ravel()[0])
                   for v in np.atleast_1d(t.values)]
        condition = dict(zip(symbols, values))
 
        # Single repeat (this file has Repeat=1). For files with >1 repeat,
        # t.repeat would be an array of mat_structs.
        reps = np.atleast_1d(t.repeat)
        repeats = []
        for r in reps:
            units = []
            for u in r.data:                   # N_units cells
                if len(u) == 0:                # empty unit slot
                    spike_times = np.array([])
                    waveforms   = np.zeros((0, 0), dtype=np.int8)
                else:
                    spike_times = np.asarray(u[0]).ravel().astype(float)  # seconds
                    waveforms   = np.asarray(u[1])                        # (samples, N_spikes)
                units.append({'spike_times': spike_times,
                              'waveforms':   waveforms})
            repeats.append({
                'tzero_us':  float(r.tzero),   # trial start time on the recording clock (microsec)
                'sync':      float(r.sync),
                'syncCount': int(r.syncCount),
                'units':     units,
            })
 
        trials.append({'trial_index': i,
                       'condition':   condition,
                       'repeats':     repeats})
 
    return {'meta': meta, 'trials': trials}
 
 
def summarize(parsed):
    m = parsed['meta']
    print(f"Animal {m['animal']}, unit {m['unit']}, hemisphere={m['hemisphere']}, "
          f"penetration={m['penetration']}")
    print(f"Display refresh: {m['refresh_hz']} Hz")
    print(f"{m['n_units']} electrodes recorded: {m['electrodes']}")
    print(f"{m['n_trials']} trials\n")
 
    # Condition matrix
    syms = list(parsed['trials'][0]['condition'].keys())
    uniq = {s: sorted({t['condition'][s] for t in parsed['trials']}) for s in syms}
    print(f"Condition variables: {syms}")
    for s, vs in uniq.items():
        print(f"  {s}: {len(vs)} unique values ({vs[0]}..{vs[-1]})")
 
    # Spike counts per unit (summed across all trials)
    n_units = m['n_units']
    counts = np.zeros(n_units, dtype=int)
    for t in parsed['trials']:
        for r in t['repeats']:
            for u_i, u in enumerate(r['units']):
                counts[u_i] += u['spike_times'].size
    print("\nTotal spikes per electrode (across all trials):")
    for elec, c in zip(m['electrodes'], counts):
        print(f"  elec{elec:3d}: {c:6d} spikes")
 
 
if __name__ == '__main__':
    file_path = "data/pvc-1/crcns-ringach-data/neurodata/ac1/ac1_u009_001.mat"
    parsed = load_pepana(file_path)
    summarize(parsed)
 
    # Example: extract spike times for electrode 5 on the first trial
    t0 = parsed['trials'][0]
    rep = t0['repeats'][0]
    print(f"\nExample — trial 0, condition {t0['condition']}:")
    print(f"  elec{parsed['meta']['electrodes'][0]} fired {rep['units'][0]['spike_times'].size} spikes")
    print(f"  first 8 spike times (s): {rep['units'][0]['spike_times'][:8]}")
