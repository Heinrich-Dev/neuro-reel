import numpy as np
from parse_data import convert_pvc1_to_simple_format
from spike_sorting import sort_spikes

FILE_PATH = './pvc-1/crcns-ringach-data/neurodata/ac1/ac1_u005_007.mat'

def align_spikes_to_stimulus(parsed_data, clean_spikes, t_delay=0.060):
    """
    Maps an array of global spike timestamps to specific movie frames.
    """
    print(f"Aligning {len(clean_spikes)} single-unit spikes to visual stimulus...")

    # Frame duration is exactly 3/90 seconds (approx 33.33ms)
    t_frame = 3.0 / 90.0 

    aligned_events = []

    # Extract trials and ensure they are sorted chronologically by tzero
    trials = [t for t in parsed_data['allTrials'] if t is not None]
    trials_sorted = sorted(trials, key=lambda x: x['tzero'])

    for spike_time in clean_spikes:
        # 1. Find which trial this global spike belongs to
        # A spike belongs to the trial with the largest tzero that is <= spike_time
        valid_trials = [t for t in trials_sorted if t['tzero'] <= spike_time]
        if not valid_trials:
            continue # Spike happened before the very first trial started

        current_trial = valid_trials[-1]

        # 2. Calculate the relative time within this specific 30-second segment
        rel_time = spike_time - current_trial['tzero']

        # 3. Account for the 60ms biological delay from retina to V1
        adjusted_time = rel_time - t_delay

        # 4. Filter out spikes triggered by whatever was on screen BEFORE the segment started
        if adjusted_time < 0:
            continue

        # 5. Calculate the zero-indexed frame number
        frame_idx = int(np.floor(adjusted_time / t_frame))

        # Each movie has about 900 frames (30s of stimulation at ~30fps). 
        # Filter out spikes that technically fall in the gap between trials.
        if frame_idx >= 900:
            continue

        aligned_events.append({
            'global_time': spike_time,
            'movie_id': current_trial['movie_id'],
            'segment_id': current_trial['segment_id'],
            'frame_idx': frame_idx
        })

    print(f"Successfully aligned {len(aligned_events)} spikes to specific frames.")
    return aligned_events

# === Example Usage ===
if __name__ == "__main__":
    # Parse the PVC-1 data
    parsed_data = convert_pvc1_to_simple_format(FILE_PATH)

    # Perform spike sorting
    clean_unit_timestamps = sort_spikes(parsed_data, channel_idx=13)

    # Assuming 'parsed_data' is loaded and 'clean_unit_timestamps' is returned from spike sorting
    aligned_spikes = align_spikes_to_stimulus(parsed_data, clean_unit_timestamps)
    
    # Example Output Inspection
    for event in aligned_spikes:
        print(f"Spike at {event['global_time']:.2f}s -> movie{event['movie_id']:03d}_{event['segment_id']:03d}_{event['frame_idx']:03d}.jpeg")
