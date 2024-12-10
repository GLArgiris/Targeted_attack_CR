import os
import numpy as np
from scipy.spatial.distance import squareform
from brainconn.distance import distance_wei
from brainconn.clustering import clustering_coef_wu
from brainconn.efficiency import charpath
from brainconn.modularity import modularity_und
from brainconn.components import get_components
from brainconn.degree import strengths_und, degrees_und

def compute_metrics_org(thresholded_matrix_path, distance_matrix_path, output_directory):
    """
    Compute graph theory metrics from thresholded and distance matrices.
    
    Parameters:
        thresholded_matrix_path (str): Path to the thresholded matrix (.npy file).
        distance_matrix_path (str): Path to the distance matrix (.npy file).
        output_directory (str): Directory to save the computed metrics.
    """
    # Load input matrices
    thresholded_matrix = np.load(thresholded_matrix_path)
    distance_matrix = np.load(distance_matrix_path)

    # Initialize arrays to store metrics
    subnum, nn = thresholded_matrix.shape[1], 420 #number of ROIs plus 1 extra for the metrics of the original unlesioned matrix
    s_lam_org = np.full((subnum, nn), np.nan)
    s_geff_org = np.full((subnum, nn), np.nan)
    s_clus_org = np.full((subnum, nn), np.nan)
    s_lcc_org = np.full((subnum, nn), np.nan)
    s_mod_org = np.full((subnum, nn), np.nan)
    nodedegree_org = np.full((subnum, nn, nn - 1), np.nan)
    nodecount_org = np.full((subnum, nn), np.nan)

    for s in range(subnum):
        print(f"Processing subject {s + 1}/{subnum}")
        # Convert matrices to square form for each subject
        tempsub1 = squareform(distance_matrix[:, s])
        tempsub2 = squareform(thresholded_matrix[:, s])

        # Original calculation
        tempdist = distance_wei(tempsub1)
        s_lam_org[s, 0], s_geff_org[s, 0] = charpath(tempdist) #in BCT for matlab, charpath has two additional arguments after matrix input, where diagonal and infinite values can be excluded using 0. Here, I think only infinite values are excluded but diagonal values (self-connections) are maintained. TO CHECK!
        s_clus_org[s, 0] = np.nanmean(clustering_coef_wu(tempsub2))
        _, s_mod_org[s, 0] = modularity_und(tempsub2)
        _, tempcomp = get_components(tempsub2)
        s_lcc_org[s, 0] = max(tempcomp)

        # Original strengths and degrees for nodes 
        tempstrength = strengths_und(tempsub2) #weight preserved and almost impossible for two nodes to have the same strength (better to use this one)
        nodedegree_org[s, 0, :] = degrees_und(tempsub2) #weight information is discarded and some nodes could have the same number
        nodecount_org[s, 0] = 0

        for i in range(1, nn):
            if i == 1:
                # Remove the node with the highest nodal strength
                tempsub1[:, np.argmax(tempstrength)] = np.inf
                tempsub1[np.argmax(tempstrength), :] = np.inf
                tempsub2[:, np.argmax(tempstrength)] = 0
                tempsub2[np.argmax(tempstrength), :] = 0
                nodecount_org[s, i] = np.argmax(tempstrength)
                nodedegree_org[s, i, :] = degrees_und(tempsub2)
            else:
                # Each iteration after the first nodal removal will have to have the nodal strength recalculated
                tempstrength = strengths_und(tempsub2) 
                ind2 = np.argmax(tempstrength)
                tempsub1[:, ind2] = np.inf
                tempsub1[ind2, :] = np.inf
                tempsub2[:, ind2] = 0
                tempsub2[ind2, :] = 0
                nodecount_org[s, i] = ind2
                nodedegree_org[s, i, :] = degrees_und(tempsub2)

            # Break if graph becomes disconnected
            if np.sum(tempsub2) == 0:
                break

            tempdist = distance_wei(tempsub1)
            s_lam_org[s, i], s_geff_org[s, i] = charpath(tempdist)
            s_clus_org[s, i] = np.nanmean(clustering_coef_wu(tempsub2))
            _, s_mod_org[s, i] = modularity_und(tempsub2)
            _, tempcomp = get_components(tempsub2)
            s_lcc_org[s, i] = max(tempcomp)

    # Save the results
    os.makedirs(output_directory, exist_ok=True)
    np.save(os.path.join(output_directory, "s_lam_org.npy"), s_lam_org)
    np.save(os.path.join(output_directory, "s_geff_org.npy"), s_geff_org)
    np.save(os.path.join(output_directory, "s_clus_org.npy"), s_clus_org)
    np.save(os.path.join(output_directory, "s_lcc_org.npy"), s_lcc_org)
    np.save(os.path.join(output_directory, "s_mod_org.npy"), s_mod_org)
    np.save(os.path.join(output_directory, "nodecount_org.npy"), nodecount_org)
    np.save(os.path.join(output_directory, "nodedegree_org.npy"), nodedegree_org)
    print(f"Metrics saved to {output_directory}")

def main():
    """
    Main function to handle arguments and execute computations.
    """
    import sys

    if len(sys.argv) != 4:
        print("Usage: python graph_theory_data_metrics.py <thresholded_matrix_path> <distance_matrix_path> <output_directory>")
        sys.exit(1)

    thresholded_matrix_path = sys.argv[1] #path to thresholded matrix (in output folder)
    distance_matrix_path = sys.argv[2] #path to distance matrix (in output folder)
    output_directory = sys.argv[3] #output directory path

    compute_metrics_org(thresholded_matrix_path, distance_matrix_path, output_directory)

if __name__ == "__main__":
    main()