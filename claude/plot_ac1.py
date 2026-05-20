"""
Plot summary figures from a parsed pepANA dictionary.
Run after parse_ac1.py is importable from the same directory.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from parse_ac1 import load_pepana

PATH = '/mnt/user-data/uploads/ac1_u004_000.mat'
parsed = load_pepana(PATH)
meta = parsed['meta']
trials = parsed['trials']
elecs = meta['electrodes']
n_units = meta['n_units']

# ---------- helpers ----------
def trial_duration(trial):
    """Use the latest spike across all units as a proxy for trial length."""
    last = 0.0
    for u in trial['repeats'][0]['units']:
        if u['spike_times'].size:
            last = max(last, float(u['spike_times'][-1]))
    return last

# Estimate a common analysis window (most trials run ~30 s based on inspection)
durations = [trial_duration(t) for t in trials]
T = float(np.median(durations))   # ≈ 30 s

# ---------- compute summary quantities ----------

# (1) Mean firing rate per unit per trial — heatmap of shape (n_units, n_trials)
rate_matrix = np.zeros((n_units, len(trials)))
for j, tr in enumerate(trials):
    dur = trial_duration(tr) or T
    for u_i, u in enumerate(tr['repeats'][0]['units']):
        rate_matrix[u_i, j] = u['spike_times'].size / dur

# (2) Pick the most active unit for a detailed raster + PSTH
total_per_unit = rate_matrix.sum(axis=1)
focus_unit = int(np.argmax(total_per_unit))
focus_elec = elecs[focus_unit]

# (3) PSTH: average across all trials for the focus unit
bin_size = 0.05  # 50 ms bins
edges = np.arange(0, T + bin_size, bin_size)
centers = 0.5 * (edges[:-1] + edges[1:])
psth_counts = np.zeros(len(centers))
n_used = 0
for tr in trials:
    st = tr['repeats'][0]['units'][focus_unit]['spike_times']
    if st.size:
        h, _ = np.histogram(st, bins=edges)
        psth_counts += h
        n_used += 1
psth_rate = psth_counts / (n_used * bin_size)  # spikes/s

# (4) Mean waveform for the focus unit (across all spikes in all trials)
wfs = [tr['repeats'][0]['units'][focus_unit]['waveforms'] for tr in trials]
wfs = [w for w in wfs if w.size]
all_wf = np.concatenate(wfs, axis=1).astype(float)  # (48, total_spikes)
wf_mean = all_wf.mean(axis=1)
wf_std  = all_wf.std(axis=1)
wf_time_ms = np.arange(48) / 24.0  # samples assumed ~24 kHz; axis is illustrative

# ---------- figure ----------
fig = plt.figure(figsize=(13, 9), constrained_layout=True)
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.1, 1])

# Panel A: heatmap of firing rate (unit × trial)
axA = fig.add_subplot(gs[0, 0])
im = axA.imshow(rate_matrix, aspect='auto', cmap='magma', origin='lower',
                extent=[0, len(trials), -0.5, n_units - 0.5])
axA.set_xlabel('Trial index (movie_id × segment_id grid)')
axA.set_ylabel('Unit (electrode)')
axA.set_yticks(range(n_units))
axA.set_yticklabels([f'e{e}' for e in elecs], fontsize=8)
axA.set_title('A. Mean firing rate per unit per trial')
cbar = fig.colorbar(im, ax=axA, label='Rate (spikes/s)')

# Panel B: spike raster for the focus unit across all trials
axB = fig.add_subplot(gs[0, 1])
for j, tr in enumerate(trials):
    st = tr['repeats'][0]['units'][focus_unit]['spike_times']
    if st.size:
        axB.vlines(st, j + 0.5, j + 1.5, color='k', linewidth=0.3)
axB.set_xlim(0, T)
axB.set_ylim(0.5, len(trials) + 0.5)
axB.invert_yaxis()
axB.set_xlabel('Time within trial (s)')
axB.set_ylabel('Trial #')
axB.set_title(f'B. Raster — electrode {focus_elec} (most active unit), all 120 trials')

# Panel C: PSTH for the focus unit (averaged across trials)
axC = fig.add_subplot(gs[1, 0])
axC.fill_between(centers, 0, psth_rate, step='mid', alpha=0.6, color='steelblue')
axC.plot(centers, psth_rate, drawstyle='steps-mid', color='navy', linewidth=0.8)
axC.set_xlim(0, T)
axC.set_xlabel('Time within trial (s)')
axC.set_ylabel('Firing rate (spikes/s)')
axC.set_title(f'C. PSTH — electrode {focus_elec}, {bin_size*1000:.0f} ms bins, '
              f'averaged over {n_used} trials')

# Panel D: mean spike waveform with ±1 SD shading
axD = fig.add_subplot(gs[1, 1])
axD.fill_between(wf_time_ms, wf_mean - wf_std, wf_mean + wf_std,
                 color='crimson', alpha=0.25, label='±1 SD')
axD.plot(wf_time_ms, wf_mean, color='crimson', linewidth=1.8, label='mean')
axD.axhline(0, color='gray', linewidth=0.5)
axD.set_xlabel('Time (ms, assuming 24 kHz)')
axD.set_ylabel('Amplitude (a.u., int8)')
axD.set_title(f'D. Mean spike waveform — electrode {focus_elec} '
              f'(n = {all_wf.shape[1]:,} spikes)')
axD.legend(loc='upper right', frameon=False)

fig.suptitle(f"{meta['animal']} unit {meta['unit']} — "
             f"{n_units} electrodes, {meta['n_trials']} trials "
             f"(movie_id × segment_id)",
             fontsize=13, fontweight='bold')

out_path = '/mnt/user-data/outputs/ac1_u004_summary.png'
fig.savefig(out_path, dpi=140, bbox_inches='tight')
print(f"Saved: {out_path}")
print(f"Focus unit was electrode {focus_elec} "
      f"(total {int(total_per_unit[focus_unit]):,} spikes).")
