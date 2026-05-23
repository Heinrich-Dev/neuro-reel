import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from parse_data import convert_pvc1_to_simple_format

FILE_PATH = './pvc-1/crcns-ringach-data/neurodata/ac1/ac1_u005_007.mat'

def sort_spikes(parsed_data, channel_idx, num_clusters=2):
    """
    Performs PCA and K-Means clustering on the waveforms of a specific channel
    to isolate single-unit activity from background noise.
    """
    print(f"--- Spike Sorting for Channel {channel_idx} ---")

    # 1. Extract and format the waveforms
    # parsed_data['allWaveforms'][channel_idx] is shape (48, num_spikes)
    # Scikit-learn expects (n_samples, n_features), so we transpose it
    waveforms = parsed_data['allWaveforms'][channel_idx].T

    num_spikes = waveforms.shape[0]
    if num_spikes < 10:
        print("Not enough spikes on this channel to cluster.")
        return

    print(f"Total threshold crossings (waveforms) to sort: {num_spikes}")

    # 2. Dimensionality Reduction (PCA)
    # Reduce the 48 voltage samples down to 2 principal components 
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(waveforms)

    # 3. K-Means Clustering
    # We expect 2 clusters: one for the neuron, one for the noise
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(principal_components)

    # Automatically identify the "Noise" cluster 
    # The noise cluster is usually the one centered closest to the origin (0,0)
    centroids = kmeans.cluster_centers_
    dist_to_origin = np.linalg.norm(centroids, axis=1)
    noise_cluster_idx = np.argmin(dist_to_origin)
    unit_cluster_idx = 1 - noise_cluster_idx # Assuming 2 clusters total

    print(f"Detected {np.sum(labels == unit_cluster_idx)} valid single-unit spikes.")
    print(f"Detected {np.sum(labels == noise_cluster_idx)} noise events.")

    # 4. Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot A: PCA Scatter Plot ---
    ax1.scatter(principal_components[labels == noise_cluster_idx, 0], 
                principal_components[labels == noise_cluster_idx, 1], 
                c='blue', alpha=0.5, label='Noise', s=5)

    ax1.scatter(principal_components[labels == unit_cluster_idx, 0], 
                principal_components[labels == unit_cluster_idx, 1], 
                c='green', alpha=0.5, label='Single-Unit', s=5)

    ax1.scatter(0, 0, c='red', marker='X', s=100, label='Origin (0,0)')
    ax1.set_title(f'PCA of Channel {channel_idx} Waveforms')
    ax1.set_xlabel('Principal Component 1')
    ax1.set_ylabel('Principal Component 2')
    ax1.legend()

    # --- Plot B: Mean Waveforms ---
    # Calculate the mean and standard deviation for each cluster
    time_pts = np.arange(48)

    mean_noise = np.mean(waveforms[labels == noise_cluster_idx], axis=0)
    std_noise = np.std(waveforms[labels == noise_cluster_idx], axis=0)

    mean_unit = np.mean(waveforms[labels == unit_cluster_idx], axis=0)
    std_unit = np.std(waveforms[labels == unit_cluster_idx], axis=0)

    ax2.plot(time_pts, mean_noise, c='blue', label='Mean Noise')
    ax2.fill_between(time_pts, mean_noise-std_noise, mean_noise+std_noise, color='blue', alpha=0.2)

    ax2.plot(time_pts, mean_unit, c='green', label='Mean Single-Unit')
    ax2.fill_between(time_pts, mean_unit-std_unit, mean_unit+std_unit, color='green', alpha=0.2)

    ax2.set_title(f'Clustered Waveform Signatures')
    ax2.set_xlabel('Sample Index (1.6ms duration)')
    ax2.set_ylabel('Voltage (\u03bcV)')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Return the clean timestamps corresponding ONLY to the real neuron
    all_spike_times = parsed_data['allSpikes'][channel_idx]
    clean_unit_timestamps = all_spike_times[labels == unit_cluster_idx]

    return clean_unit_timestamps

# === How to run it ===
if __name__ == "__main__":
    # Assuming parsed_data is in memory from your previous script
    parsed_data = convert_pvc1_to_simple_format(FILE_PATH)

    # Pick a channel to analyze (e.g., channel 13 is mentioned in the docs as having good spikes)
    target_channel = 13

    # Ensure the channel index is within bounds for this specific file
    if target_channel < parsed_data['numChannels']:
        clean_spikes = sort_spikes(parsed_data, channel_idx=target_channel)
