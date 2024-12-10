import numpy as np

def calculate_small_world_index(s_clus_org, s_clus_rand, s_lam_org, s_lam_rand):
    """
    Calculate the small-world network index for each subject and time point.

    Args:
        s_clus_org (ndarray): The global clustering coefficient of the original data matrix (participants x nodal iterations).
        s_clus_rand (ndarray): The global clustering coefficient matrix of the randomized matrix (participants x nodal iterations).
        s_lam_org (ndarray): The characteristic path length of the original data matrix (participants x nodal iterations).
        s_lam_rand (ndarray): The characteristic path length of the randomized matrix (participants x nodal iterations).
    Returns:
        SD_sw_long (ndarray): The small-world index matrix (subjects x time points).
    """
    # Initialize the SD_sw_long matrix with NaN values
    SD_sw_long = np.full(s_clus_org.shape, np.nan)

    # Calculate the small-world network index for each subject and time point
    for s in range(s_clus_org.shape[0]):  # Iterate over subjects
        for i in range(s_clus_org.shape[1]):  # Iterate over time points
            # Calculate the small-world index for each subject and time point
            SD_sw_long[s, i] = (s_clus_org[s, i] / s_clus_rand[s, i]) / (s_lam_org[s, i] / s_lam_rand[s, i])

    return SD_sw_long

def main():
    """
    Main function to load matrices and calculate the small-world network index.
    """
    # Load the matrices from the output folder (replace with correct file paths)
    s_clus_org = np.load("s_clus_org.npy")  # Load observed clustering coefficient
    s_clus_rand = np.load("s_clus_rand.npy")  # Load random clustering coefficient
    s_lam_org = np.load("s_lam_org.npy")  # Load observed path length
    s_lam_rand = np.load("s_lam_rand.npy")  # Load random path length

    # Calculate the small-world network index
    SD_sw_long = calculate_small_world_index(s_clus_org, s_clus_rand, s_lam_org, s_lam_rand)

    # Save the small-world network index matrix
    np.save("SD_sw_long.npy", SD_sw_long)
    print("Small-world network index calculated and saved.")

if __name__ == "__main__":
    main()
