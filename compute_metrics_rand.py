import numpy as np
from brainconn.efficiency import charpath
from brainconn.clustering import clustering_coef_wu
from brainconn.modularity import modularity_und
from brainconn.distance import distance_wei
from brainconn.random_graphs import randmio_und
from scipy.io import loadmat, savemat
from scipy.spatial.distance import squareform

def compute_metrics_rand(input_distance_matrix, input_threshold_matrix, output_file):
    """
    Compute graph theory metrics on random matrices derived from the thresholded and distance matrices.
    Parameters:
        thresholded_matrix_path (str): Path to the thresholded matrix (.npy file).
        distance_matrix_path (str): Path to the distance matrix (.npy file).
        output_directory (str): Directory to save the computed metrics.
    """
    # Load input matrices
    thresholded_matrix = np.load(thresholded_matrix_path)
    distance_matrix = np.load(distance_matrix_path)

    # Initialize variables
    subnum, nn = thresholded_matrix.shape[1], 420 #number of ROIs plus 1 extra for the 
    s_lam_rand = np.full((subnum, nn), np.nan)
    s_geff_rand = np.full((subnum, nn), np.nan)
    s_clus_rand = np.full((subnum, nn), np.nan)
    s_lcc_rand = np.full((subnum, nn), np.nan)
    s_mod_rand = np.full((subnum, nn), np.nan)

    # Loop through subjects
    for s in range(subnum):
        print(f"Processing subject {s + 1}/{subnum}...")
        original_matrix = squareform(thresholded_matrix[:, s])  # Convert vectorized form to matrix
        random_matrix = randmio_und(original_matrix, 10)  # Randomize the graph while preserving degree and specifying the number of times you want to randomize. I chose 10. 
        random_dist_matrix = 1 - random_matrix  # Create a "distance" matrix

        # Replace diagonal with Inf (or 0 as needed)
        np.fill_diagonal(random_dist_matrix, np.inf)

                # Compute metrics iteratively
        for i in range(nn - 1):
            # Remove highest-strength node using strengths_und
            temp_strength = strengths_und(random_matrix)
            highest_strength_node = np.argmax(temp_strength)
            random_matrix[:, highest_strength_node] = 0
            random_matrix[highest_strength_node, :] = 0
            random_dist_matrix[:, highest_strength_node] = np.inf
            random_dist_matrix[highest_strength_node, :] = np.inf

            if np.sum(random_matrix) > 0:
                temp_dist = distance_wei(random_dist_matrix)

                # Directly store metrics
                s_lam_rand[s, i], s_geff_rand[s, i] = charpath(temp_dist)  # Characteristic path length (lambda) and global efficiency
                s_clus_rand[s, i] = np.nanmean(clustering_coef_wu(random_matrix)) 
                s_mod_rand[s, i], _ = modularity_und(random_matrix) 
                components = np.unique(random_matrix)
                s_lcc_rand[s, i] = max(components)  # Largest connected component
            else:
                break

    # Save results
    savemat(output_file, {
        's_lam_rand': s_lam_rand,
        's_geff_rand': s_geff_rand,
        's_clus_rand': s_clus_rand,
        's_lcc_rand': s_lcc_rand,
        's_mod_rand': s_mod_rand
    })
    print(f"Random metrics saved to {output_file}.")

def main():
    """
    Main function to handle arguments and execute computations.
    """
    if len(sys.argv) != 4:
        print("Usage: python graph_theory_data_metrics.py <thresholded_matrix_path> <distance_matrix_path> <output_file>")
        sys.exit(1)

    thresholded_matrix_path = sys.argv[1] #path to thresholded matrix (in output folder)
    distance_matrix_path = sys.argv[2] #path to distance matrix (in output folder)
    output_file = sys.argv[3] #output directory path

    compute_metrics_rand(thresholded_matrix_path, distance_matrix_path, output_file)

if __name__ == "__main__":
    main()