"""
Loader for the Allen Brain Observatory — Visual Coding Neuropixels dataset.

Reads one ecephys session (mouse V1 + higher visual areas) and returns a
`Session` conforming to loaders.base. Currently focuses on natural_movie_one
and natural_movie_three stimuli, which align best with the natural-movie
prediction task this codebase is built around.

==============================================================================
                                  CAVEAT
==============================================================================
This loader was written without a live `allensdk` install available for
verification — the package failed to build on Python 3.12 in the dev
environment. The structure follows the documented SDK patterns but specific
method names, return shapes, and field names may need correction when run
against the real SDK.

Three things to check first if it doesn't work out of the box:

  1. The import path. `EcephysProjectCache` lives in
     `allensdk.brain_observatory.ecephys.ecephys_project_cache` in older
     SDK versions; newer versions may have moved it. If the import fails,
     check `allensdk.brain_observatory.ecephys.behavior_ecephys_session`
     and `abc_atlas_access` (the successor package for some products).

  2. The session-loading method. I use `cache.get_session_data(session_id)`.
     Some SDK versions call this `get_session` or require an explicit
     `EcephysSession.from_nwb_path()` call instead.

  3. Stimulus accessor methods. I assume `session.stimulus_presentations`
     returns a DataFrame and `session.get_stimulus_template(name)` returns
     the movie array. Both have existed for a long time but the exact
     column names in stimulus_presentations have shifted ('stimulus_name'
     vs 'stimulus_block', frame indexing schemes etc).

Use this file as a structural template and adjust SDK-specific calls
against the real API. Schema output is correct; SDK plumbing may need
small fixes.
==============================================================================

Recommended starting recipe:
  - filter session_table to 'brain_observatory_1.1' protocol sessions
  - within each session, filter units by Allen's recommended QC thresholds
    (isi_violations < 0.5, presence_ratio > 0.9, amplitude_cutoff < 0.1)
  - load natural_movie_one (30 s) as the closest analog to a pvc-1 segment
"""
import os
from typing import Optional

import numpy as np

from .base import Trial, Session


# Allen natural movies are 30 fps, same as pvc-1 — no rebinning needed.
DEFAULT_FRAME_RATE_HZ = 30.0

# Recommended Allen QC thresholds for "good" units (from their tutorials).
DEFAULT_QC = {
    "isi_violations":   0.5,
    "presence_ratio":   0.9,
    "amplitude_cutoff": 0.1,
}


def _try_import_allensdk():
    """Defer the allensdk import so the loader file can be imported without
    the SDK installed. Returns (EcephysProjectCache, error_or_None)."""
    try:
        from allensdk.brain_observatory.ecephys.ecephys_project_cache \
            import EcephysProjectCache
        return EcephysProjectCache, None
    except ImportError as e:
        return None, e


def _filter_units(units_df, qc: dict, area: str):
    """Apply QC and brain-area filtering. Returns a sub-DataFrame."""
    # Numeric QC thresholds. Units missing any column are kept (NaN < threshold
    # is False, but we want to keep rather than discard for safety).
    keep = np.ones(len(units_df), dtype=bool)
    if "isi_violations" in units_df.columns:
        keep &= (units_df["isi_violations"].fillna(0) < qc["isi_violations"])
    if "presence_ratio" in units_df.columns:
        keep &= (units_df["presence_ratio"].fillna(1) > qc["presence_ratio"])
    if "amplitude_cutoff" in units_df.columns:
        keep &= (units_df["amplitude_cutoff"].fillna(0) < qc["amplitude_cutoff"])

    # Area filter. The 'ecephys_structure_acronym' column is the Allen-standard
    # location label; 'VISp' is primary visual cortex.
    if area is not None and "ecephys_structure_acronym" in units_df.columns:
        keep &= (units_df["ecephys_structure_acronym"] == area)

    return units_df[keep]


def _resize_movie(movie: np.ndarray, target_hw: Optional[tuple]) -> np.ndarray:
    """Resize an Allen stimulus template to a target (H, W). No-op if None."""
    if target_hw is None:
        return movie
    from PIL import Image
    H, W = target_hw
    out = np.empty((movie.shape[0], H, W), dtype=movie.dtype)
    for i in range(movie.shape[0]):
        im = Image.fromarray(movie[i]).resize((W, H), Image.BILINEAR)
        out[i] = np.asarray(im)
    return out


def _make_frames_provider(movie_array: np.ndarray):
    """Return a callable matching the schema's frames_provider contract.

    Allen natural movies are stored as a single 3D array per stimulus
    template. Different presentations are sub-clips indexed by start_frame.
    The frames_provider receives a stimulus_key of (movie_name, start, end)
    and returns the sub-clip frames.
    """
    def provider(stimulus_key):
        movie_name, start_frame, end_frame = stimulus_key
        # movie_array is bound at provider-creation time; we ignore movie_name
        # here because one provider corresponds to one loaded movie. The
        # session-level coordination is handled in load_allen_session().
        return movie_array[start_frame:end_frame]
    return provider


def load_allen_session(
    session_id: int,
    manifest_path: str,
    stimulus_name: str = "natural_movie_one",
    area: str = "VISp",
    qc: Optional[dict] = None,
    target_hw: Optional[tuple] = (240, 320),
    frame_rate_hz: float = DEFAULT_FRAME_RATE_HZ,
) -> Session:
    """Load one Allen ecephys session as a `Session` object.

    Parameters
    ----------
    session_id : int
        The Allen session identifier (column 'id' in cache.get_session_table()).
    manifest_path : str
        Path to the SDK cache manifest. The first call downloads the session
        NWB file (~2-3 GB) into the cache directory derived from this path.
    stimulus_name : str
        Which stimulus to extract. 'natural_movie_one' (30 s clip, ~10 reps
        per session) or 'natural_movie_three' (120 s clip, fewer reps) are
        the natural-movie options. Other Allen stimuli need different handling.
    area : str
        Brain-area filter. 'VISp' = primary visual cortex. Pass None to keep
        all areas.
    qc : dict or None
        Override the default QC thresholds (see DEFAULT_QC at module top).
    target_hw : (H, W) or None
        Resize movie frames to this shape at load time. The Allen natural
        movies are ~304x608; resize to (240, 320) to align with the pvc-1
        model input. Pass None to keep native resolution.
    frame_rate_hz : float
        Override movie playback rate. Default 30 Hz matches Allen's natural
        movies.

    Returns
    -------
    Session
        Conforming to loaders.base.Session. One Trial per stimulus
        presentation (so one session typically yields 10-30 trials, vs.
        pvc-1's 120 unique trials, but each trial has internal repeats
        that can be exploited downstream for trial-averaged training).
    """
    EcephysProjectCache, import_err = _try_import_allensdk()
    if EcephysProjectCache is None:
        raise ImportError(
            f"allensdk not available: {import_err}\n"
            "Install with: pip install allensdk\n"
            "Note: allensdk has historically required Python <=3.10."
        ) from import_err

    if qc is None:
        qc = DEFAULT_QC

    # ---- Open cache, load session ----
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    session = cache.get_session_data(session_id)
    # ^ NOTE: see CAVEAT in module docstring. Method name may need adjustment.

    # ---- Filter units by QC and area ----
    units_df = _filter_units(session.units, qc=qc, area=area)
    if len(units_df) == 0:
        raise ValueError(
            f"No units passed QC + area={area} filter in session {session_id}. "
            "Relax QC thresholds or pass area=None.")

    unit_ids = [str(uid) for uid in units_df.index.tolist()]
    n_units = len(unit_ids)

    # Per-unit metadata for the schema. Allen ships a rich set of columns;
    # we keep a useful subset and let consumers reach into the raw DataFrame
    # via Session.metadata for anything else.
    keep_cols = [c for c in
                 ["ecephys_structure_acronym", "probe_horizontal_position",
                  "probe_vertical_position", "snr", "firing_rate",
                  "isi_violations", "presence_ratio", "amplitude_cutoff",
                  "waveform_duration"]
                 if c in units_df.columns]
    unit_metadata = [
        {**{c: row[c] for c in keep_cols}, "unit_id": uid}
        for uid, row in zip(unit_ids, units_df.to_dict("records"))
    ]

    # ---- Load the stimulus template (one big movie array) ----
    movie = session.get_stimulus_template(stimulus_name)
    # ^ Expected shape: (n_frames, H, W); dtype uint8 grayscale
    if movie.ndim == 4 and movie.shape[-1] == 3:
        # RGB -> grayscale via ITU-R BT.601 weights
        movie = (0.299 * movie[..., 0]
                 + 0.587 * movie[..., 1]
                 + 0.114 * movie[..., 2]).astype(np.uint8)
    movie = _resize_movie(movie, target_hw)
    native_shape = (int(movie.shape[1]), int(movie.shape[2]))

    # ---- Build trials from stimulus_presentations ----
    presentations = session.stimulus_presentations
    presentations = presentations[presentations["stimulus_name"] == stimulus_name]
    if len(presentations) == 0:
        raise ValueError(
            f"No '{stimulus_name}' presentations found in session {session_id}")

    # Allen presentations are typically per-FRAME of the movie (one row per
    # frame index). We need to group consecutive frames into a single "trial"
    # representing one full clip presentation. The 'stimulus_block' column
    # (when present) marks each repetition.
    if "stimulus_block" in presentations.columns:
        groups = presentations.groupby("stimulus_block")
    else:
        # Fallback: detect repeats by frame_index wrapping back to 0
        frame_col = ("frame" if "frame" in presentations.columns
                     else "stimulus_index")
        block_id = (presentations[frame_col].diff().fillna(1) <= 0).cumsum()
        groups = presentations.groupby(block_id)

    # Pre-fetch spike times once; slicing them per-trial is cheap.
    all_spike_times = {uid: np.asarray(session.spike_times[int(uid)])
                       for uid in unit_ids}

    trials = []
    for block_id, block in groups:
        t_start = float(block["start_time"].iloc[0])
        t_stop  = float(block["stop_time"].iloc[-1])
        frame_col = ("frame" if "frame" in block.columns else "stimulus_index")
        start_frame = int(block[frame_col].iloc[0])
        end_frame   = int(block[frame_col].iloc[-1]) + 1

        # Trial-relative spike times for each unit
        spike_times = []
        for uid in unit_ids:
            st = all_spike_times[uid]
            mask = (st >= t_start) & (st < t_stop)
            spike_times.append((st[mask] - t_start).astype(np.float64))

        trials.append(Trial(
            trial_id     = f"allen_{session_id}_b{int(block_id):03d}",
            stimulus_key = (stimulus_name, start_frame, end_frame),
            spike_times  = spike_times,
            duration_s   = t_stop - t_start,
            metadata     = {"block_id": int(block_id),
                            "t_start": t_start, "t_stop": t_stop},
        ))

    # ---- Frames provider ----
    # All trials in this session use the same stimulus template; the
    # frames_provider closes over `movie` and slices it by (start, end).
    frames_provider = _make_frames_provider(movie)

    return Session(
        session_id      = f"allen_{session_id}_{stimulus_name}",
        source          = "allen-neuropixels",
        stimulus_type   = "natural_movie",
        n_units         = n_units,
        unit_ids        = unit_ids,
        unit_metadata   = unit_metadata,
        frame_rate_hz   = frame_rate_hz,
        frame_shape     = native_shape,
        frames_provider = frames_provider,
        trials          = trials,
        metadata        = {
            "allen_session_id": session_id,
            "stimulus_name":    stimulus_name,
            "area_filter":      area,
            "qc_thresholds":    dict(qc),
            "n_units_pre_qc":   int(len(session.units)),
            "n_units_post_qc":  n_units,
        },
    )


def discover_allen_sessions(
    manifest_path: str,
    min_visp_units: int = 30,
    protocol: str = "brain_observatory_1.1",
) -> list:
    """Find Allen sessions worth loading.

    Returns a list of session IDs satisfying:
      - matching session_type / protocol
      - at least `min_visp_units` units in primary visual cortex

    Useful as a pre-filter before calling load_allen_session() repeatedly.
    Note: returning a session ID here only commits to *metadata* lookup;
    the heavy NWB download happens at load time.
    """
    EcephysProjectCache, import_err = _try_import_allensdk()
    if EcephysProjectCache is None:
        raise ImportError(f"allensdk not available: {import_err}") from import_err

    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    sessions = cache.get_session_table()
    units = cache.get_unit_table()

    # Filter sessions by protocol if a protocol column exists
    if "session_type" in sessions.columns and protocol is not None:
        sessions = sessions[sessions["session_type"] == protocol]

    # Count VISp units per session
    if "ecephys_structure_acronym" in units.columns:
        visp = units[units["ecephys_structure_acronym"] == "VISp"]
        counts = visp.groupby("ecephys_session_id").size()
        good = counts[counts >= min_visp_units].index.tolist()
        return [s for s in sessions.index.tolist() if s in good]

    return sessions.index.tolist()
