"""
Precompute baseline statistics from a reference pepANA .mat file.
Run once. Produces static/baseline.json containing:
    - electrodes: list[int]
    - n_electrodes: int
    - mean_rate_hz: list[float]      per-electrode mean firing rate
    - std_rate_hz:  list[float]      across-trial std of firing rate
    - waveform_mean: list[list[float]]   (n_elec, n_samples)
    - waveform_std:  list[list[float]]   (n_elec, n_samples)
    - waveform_sample_rate_hz: float
    - n_trials: int
"""
import json
import os
import sys
import numpy as np

# Make the lib/ helpers importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))
from v1_encoder import load_session

MAT_PATH = os.path.join(os.path.dirname(__file__), 'ac1_u004_000.mat')
OUT_PATH = os.path.join(os.path.dirname(__file__), 'static', 'baseline.json')
ASSUMED_SAMPLE_RATE_HZ = 24000.0  # placeholder; not stored in .mat


def main():
    print(f"Loading {MAT_PATH} ...")
    conditions = load_session(MAT_PATH)
    n_neurons = len(conditions[0]['spike_times'])
    print(f"  {len(conditions)} trials, {n_neurons} electrodes")

    # Per-trial firing rate per electrode
    # Trial duration ≈ max spike time in that trial (proxy)
    rates = np.zeros((len(conditions), n_neurons))
    for ti, cond in enumerate(conditions):
        durs = []
        for st in cond['spike_times']:
            if st.size:
                durs.append(float(st[-1]))
        dur = max(durs) if durs else 30.0
        for ni, st in enumerate(cond['spike_times']):
            rates[ti, ni] = st.size / dur

    mean_rate = rates.mean(axis=0)
    std_rate  = rates.std(axis=0)
    print(f"  per-electrode mean rate (Hz): {np.round(mean_rate, 2).tolist()}")

    # Waveforms: need to reload with full struct access to get waveform snippets
    import scipy.io as sio
    raw = sio.loadmat(MAT_PATH, squeeze_me=True, struct_as_record=False)
    pep = raw['pepANA']
    electrodes = np.asarray(pep.elec_list, dtype=int).ravel().tolist()

    # Accumulate waveforms per electrode across all trials
    wf_per_elec = [[] for _ in range(n_neurons)]
    for cond in np.atleast_1d(pep.listOfResults):
        rep = np.atleast_1d(cond.repeat)[0]
        for ni, ch in enumerate(np.atleast_1d(rep.data)):
            if hasattr(ch, '__len__') and len(ch) >= 2:
                wf = np.asarray(ch[1])
                if wf.ndim == 2 and wf.size > 0:
                    wf_per_elec[ni].append(wf.astype(np.float32))

    wf_mean, wf_std = [], []
    for ni in range(n_neurons):
        if wf_per_elec[ni]:
            all_wf = np.concatenate(wf_per_elec[ni], axis=1)  # (samples, n_spikes)
            wf_mean.append(all_wf.mean(axis=1).tolist())
            wf_std.append(all_wf.std(axis=1).tolist())
        else:
            wf_mean.append([0.0] * 48)
            wf_std.append([0.0] * 48)
        print(f"  e{electrodes[ni]:3d}: "
              f"{sum(w.shape[1] for w in wf_per_elec[ni]):>7d} spikes")

    out = {
        'electrodes':              electrodes,
        'n_electrodes':            n_neurons,
        'mean_rate_hz':            mean_rate.tolist(),
        'std_rate_hz':             std_rate.tolist(),
        'waveform_mean':           wf_mean,
        'waveform_std':            wf_std,
        'waveform_sample_rate_hz': ASSUMED_SAMPLE_RATE_HZ,
        'n_trials':                len(conditions),
        'source':                  os.path.basename(MAT_PATH),
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f)
    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"\nWrote {OUT_PATH} ({size_kb:.1f} KB)")


if __name__ == '__main__':
    main()
