"""This module provides functionality for parsing and converting MATLAB data into Python format."""
import scipy.io as sio
import numpy as np

FILE_PATH = './pvc-1/crcns-ringach-data/neurodata/ac1/ac1_u005_007.mat'

def convert_pvc1_to_simple_format(file_path):
    """Converts a PVC-1 MATLAB file to a simple Python format.

    This function loads a MATLAB file containing PVC-1 data and converts it into
    a more accessible Python dictionary format. It extracts spike times, waveforms,
    and trial information, organizing them by channel and trial.

    Args:
        file_path (str): The path to the PVC-1 MATLAB file to be converted.

    Raises:
        ValueError: If the MATLAB file does not contain the expected structure or data.
        ValueError: If the expected symbols 'movie_id' and 'segment_id' are not found in the data.

    Returns:
        dict: A dictionary containing the converted PVC-1 data.
    """
    mat_contents = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)

    # The data root is typically contained within the 'pepANA' structure matrix
    pvc1 = mat_contents['pepANA']

    sf = {}

    # Extract metadata properties similar to pvc1_info.m
    repeats = np.atleast_1d(pvc1.listOfResults[0].repeat)
    sf['starting_time'] = repeats[0].tzero
    sf['sampling_rate'] = 30000  # 30khz sampling rate
    sf['numRepeats'] = pvc1.Repeat
    sf['numConditions'] = pvc1.no_conditions

    # Ensure elec_list is iterable (can happen if squeezed size is 1)
    elec_list = np.atleast_1d(pvc1.elec_list)
    sf['numChannels'] = len(elec_list)
    sf['elect_list'] = elec_list

    # Initialize lists to store trials, spikes, and waveforms per channel
    sf['allSpikes'] = [[] for _ in range(sf['numChannels'])]
    sf['allWaveforms'] = [[] for _ in range(sf['numChannels'])]

    num_trials = sf['numRepeats'] * sf['numConditions']
    sf['allTrials'] = [None] * num_trials

    trial_idx = 0

    # Loop over all repeats and conditions to accumulate time sequentially
    for repeat in range(sf['numRepeats']):
        for condition in range(sf['numConditions']):
            result = pvc1.listOfResults[condition]

            symbols = list(result.symbols)
            if symbols[0] not in ('movie_id', 'r') or symbols[1] not in ('segment_id', 's'):
                raise ValueError(f"Condition {condition}, repeat {repeat}, "
                                 "Expected symbols movie_id/r, segment_id/s, "
                                 "found {symbols[0]}, {symbols[1]}")

            seg_data = np.atleast_1d(result.repeat)[repeat]
            tzero = seg_data.tzero

            # 1. Define safe_values using result.values
            safe_values = np.atleast_1d(result.values)

            sf['allTrials'][trial_idx] = {
                'tzero': tzero,
                'timeTag': seg_data.timeTag,
                'movie_id': safe_values[0],
                'segment_id': safe_values[1] 
                            if len(safe_values) > 1 else None, # if segment_id is also used
            }
            trial_idx += 1

            data = seg_data.data
            for channel in range(sf['numChannels']):
                chan_data = data[channel]

                # Shift spike timings using 'tzero'
                spikes = np.atleast_1d(chan_data[0]) + tzero
                if len(spikes) > 0:
                    sf['allSpikes'][channel].append(spikes)

                waveforms = chan_data[1]
                if len(spikes) > 0 and waveforms.size > 0:
                    # ensure geometry matches even for 1 spike (48, 1)
                    waveforms = np.atleast_2d(waveforms)
                    if waveforms.shape == (48,):
                        waveforms = waveforms[:, np.newaxis]
                    sf['allWaveforms'][channel].append(waveforms)

    # Re-assemble the fragmented arrays into continuous contiguous arrays
    for channel in range(sf['numChannels']):
        if len(sf['allSpikes'][channel]) > 0:
            sf['allSpikes'][channel] = np.concatenate(sf['allSpikes'][channel])

            # Determine expected rows (e.g., 48)
            waveform_list = sf['allWaveforms'][channel]
            expected_rows = waveform_list[0].shape[0] if len(waveform_list) > 0 else 48

            # Keep only items matching expected_rows
            valid_waveforms = [w for w in waveform_list 
                               if w.ndim > 0 and w.shape[0] == expected_rows]

            if valid_waveforms:
                sf['allWaveforms'][channel] = np.concatenate(valid_waveforms, axis=1)
            else:
                sf['allWaveforms'][channel] = np.empty((expected_rows, 0))

            # Ensure chronological order validation
            if not np.all(np.diff(sf['allSpikes'][channel]) >= 0):
                raise ValueError(f"Spike times not sequential for channel {channel}")
        else:
            sf['allSpikes'][channel] = np.array([])
            sf['allWaveforms'][channel] = np.empty((48, 0))

    return sf

if __name__ == "__main__":
    parsed_data = convert_pvc1_to_simple_format(FILE_PATH)
    print("Conversion successful!")
    print(f"Total Channels: {parsed_data['numChannels']}")
    for ch in range(parsed_data['numChannels']):
        print(f"Channel {ch} spikes: {len(parsed_data['allSpikes'][ch])}")
        print(f"Channel {ch} waveforms: {parsed_data['allWaveforms'][ch].shape}")
