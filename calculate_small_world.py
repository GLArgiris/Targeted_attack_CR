import os
import sys
import numpy as np

def calculate_small_world_index(metric_data_path, output_directory):
    """
    Calculate the small-world network index for each participant and nodal iteration

    Arguments
    
        metric_data_path: file path where matrices are located
        output_directory: output directory where to save
    

    Files that it needs
        s_clus_org (ndarray): The global clustering coefficient of the original data matrix (participants x nodal iterations).
        s_clus_rand (ndarray): The global clustering coefficient matrix of the randomized matrix (participants x nodal iterations).
        s_lam_org (ndarray): The characteristic path length of the original data matrix (participants x nodal iterations).
        s_lam_rand (ndarray): The characteristic path length of the randomized matrix (participants x nodal iterations).
    Returns:
        SD_sw_long (ndarray): The small-world index matrix (participants x nodal iterations).
    """
    
    # load the files
    s_clus_org = np.load(os.path.join(metric_data_path, 's_clus_org.npy'))
    s_clus_rand = np.load(os.path.join(metric_data_path, 's_clus_rand.npy'))
    s_lam_org = np.load(os.path.join(metric_data_path, 's_lam_org.npy'))
    s_lam_rand = np.load(os.path.join(metric_data_path, 's_lam_rand.npy'))
    
    # Initialize the SD_sw_long matrix with NaN values
    SD_sw_long = np.full(s_clus_org.shape, np.nan)

    # Calculate the small-world network index for each subject and time point
    for s in range(s_clus_org.shape[0]):  # Iterate over subjects
        for i in range(s_clus_org.shape[1]):  # Iterate over time points
            # Calculate the small-world index for each subject and time point
            SD_sw_long[s, i] = (s_clus_org[s, i] / s_clus_rand[s, i]) / (s_lam_org[s, i] / s_lam_rand[s, i])

    return SD_sw_long
    
    # Save the results
    os.makedirs(output_directory, exist_ok=True)
    np.save(os.path.join(output_directory, "SD_sw_long.npy"), SD_sw_long)
    print("Small-world metric calculated and saved.")
    
def main():
    
    if len(sys.argv) != 3:
        print("Usage: python calculate_small_world.py <metric_data_path> <output_file>")
        sys.exit(1)

    metric_data_path = sys.argv[1] #path to thresholded matrix (in output folder)
    output_directory = sys.argv[2] #output directory path

    compute_metrics_rand(metric_data_path, output_directory)

if __name__ == "__main__":
    main()