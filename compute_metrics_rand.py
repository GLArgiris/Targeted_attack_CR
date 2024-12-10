import sys
import os
import numpy as np
from brainconn.distance import charpath
from brainconn.clustering import clustering_coef_wu
from brainconn.modularity import modularity_und
from brainconn.distance import distance_wei
from brainconn.degree import strengths_und
from brainconn.clustering import get_components
import networkx as nx
from scipy.io import loadmat, savemat
from scipy.spatial.distance import squareform

def compute_metrics_rand(preprocessed_data_path, output_file):
    """
    Compute graph theory metrics on random matrices derived from the thresholded and distance matrices.
    
    Arguments
        preprocessed_data_path (str): Path to the thresholded and distance matrices (.npy file).
        output_directory (str): Directory to save the computed metrics.
    """
    # Load input matrices
    thresholded_matrix = np.load(os.path.join(preprocessed_data_path, 'thresholded_matrix.npy'))
    distance_matrix = np.load(os.path.join(preprocessed_data_path, 'distance_matrix.npy'))

    # Initialize variables
    subnum, nn = thresholded_matrix.shape[1], 420 #number of ROIs plus 1 extra for the 
    s_lam_rand = np.full((subnum, nn), np.nan)
    s_geff_rand = np.full((subnum, nn), np.nan)
    s_clus_rand = np.full((subnum, nn), np.nan)
    s_lcc_rand = np.full((subnum, nn), np.nan)
    s_mod_rand = np.full((subnum, nn), np.nan)

    def randomize_graph(matrix, num_swaps=10):
        """
        Randomizes an undirected graph while preserving its degree distribution.

        Parameters:
            matrix (np.ndarray): Input adjacency matrix (undirected).
            num_swaps (int): Number of double-edge swaps to perform.

        Returns:
            np.ndarray: Randomized adjacency matrix.
        """
        # Convert the adjacency matrix to a networkx graph
        graph = nx.from_numpy_array(matrix)

        # Perform double-edge swaps to randomize edges
        nx.double_edge_swap(graph, nswap=num_swaps)

        # Convert the graph back to an adjacency matrix
        randomized_matrix = nx.to_numpy_array(graph)

        return randomized_matrix

    # Loop through subjects
    for s in range(subnum):
        print(f"Processing subject {s + 1}/{subnum}...")
        original_matrix = squareform(thresholded_matrix[:, s])  # Convert vectorized form to matrix
        random_matrix = randomize_graph(original_matrix, num_swaps=10)  # Randomize the graph while preserving degree and specifying the number of times you want to randomize. I chose 10. 
        random_dist_matrix = 1 - random_matrix  # Create a "distance" matrix

        # Replace diagonal with Inf (or 0 as needed)
        np.fill_diagonal(random_dist_matrix, np.inf)

                # Compute metrics iteratively
        for i in range(nn - 1):
            # Remove highest-strength node using strengths_und
            tempstrength = strengths_und(random_matrix)
            highest_strength_node = np.argmax(tempstrength)
            random_matrix[:, highest_strength_node] = 0
            random_matrix[highest_strength_node, :] = 0
            random_dist_matrix[:, highest_strength_node] = np.inf
            random_dist_matrix[highest_strength_node, :] = np.inf

            if np.sum(random_matrix) > 0:
                tempdist, _ = distance_wei(random_dist_matrix)

                # Directly store metrics
                s_lam_rand[s, i], s_geff_rand[s, i], *_ = charpath(tempdist,include_diagonal=False, include_infinite=False)  # Characteristic path length (lambda) and global efficiency
                s_clus_rand[s, i] = np.nanmean(clustering_coef_wu(random_matrix)) 
                _, s_mod_rand[s, i] = modularity_und(random_matrix) 
                _, tempcomp = get_components(random_matrix)
                s_lcc_rand[s, i] = max(tempcomp)  # Largest connected component
            else:
                break
            print(f"This is iteration {i + 1}")
                
    # Save the results
    os.makedirs(output_directory, exist_ok=True)
    np.save(os.path.join(output_directory, "s_lam_rand.npy"), s_lam_rand)
    np.save(os.path.join(output_directory, "s_geff_rand.npy"), s_geff_rand)
    np.save(os.path.join(output_directory, "s_clus_reand.npy"), s_clus_rand)
    np.save(os.path.join(output_directory, "s_lcc_rand.npy"), s_lcc_rand)
    np.save(os.path.join(output_directory, "s_mod_rand.npy"), s_mod_rand)
    print(f"Metrics saved to {output_directory}")

def main():
    
    if len(sys.argv) != 3:
        print("Usage: python compute_metrics_rand.py <preprocessed_data_path> <output_directory>")
        sys.exit(1)

    preprocessed_data_path = sys.argv[1] #path to thresholded matrix (in output folder)
    output_directory = sys.argv[2] #output directory path

    compute_metrics_rand(preprocessed_data_path, output_directory)

if __name__ == "__main__":
    main()