import pandas as pd
import numpy as np
import os
import sys
from scipy.spatial.distance import squareform  # Ensure you import squareform

def load_csv(input_directory, input_file_name):
    """
    Load a connectivity matrix from a CSV file, which are Fisher Z-values.

    Parameters:
        input_directory (str): Path to the input directory.
        input_file_name (str): The name of the CSV file to load.

    Returns:
        np.ndarray: The loaded matrix as a NumPy array.
    """
    file_path = os.path.join(input_directory, input_file_name)  # Combine directory and file name
    try:
        data = pd.read_csv(file_path)
        matrix = data.values  # Convert to NumPy array
        print(f"Matrix loaded successfully from {file_path}. Shape: {matrix.shape}")  # CHECK SHAPE
        return matrix
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        raise

def convert_to_r_correlation(matrix):
    """
    Convert the input matrix to r-correlation if the matrix is in Fisher Z.

    Parameters:
        matrix (np.ndarray): The input matrix with connectivity values.

    Returns:
        np.ndarray: Matrix transformed to r-correlation values.
    """
    r_matrix = (np.exp(2 * matrix) - 1) / (np.exp(2 * matrix) + 1)
    print("Converted matrix to r-correlation.")
    return r_matrix

def threshold_matrix(r_matrix, percentile):
    """
    Threshold the matrix at a given percentile.

    Parameters:
        r_matrix (np.ndarray): The r-correlation matrix.
        percentile (float): The percentile threshold.

    Returns:
        np.ndarray: The thresholded matrix.
    """
    thresholded_matrix = np.zeros_like(r_matrix)
    for col in range(r_matrix.shape[1]):  # Loop over subjects (columns)
        threshold = np.percentile(r_matrix[:, col], percentile)
        thresholded_matrix[:, col] = np.where(r_matrix[:, col] > threshold, r_matrix[:, col], 0)
    print(f"Applied {percentile}th percentile threshold.")
    return thresholded_matrix

def compute_distance_matrix(thresholded_matrix):
    """
    Compute a distance matrix from the thresholded matrix.

    Parameters:
        thresholded_matrix (np.ndarray): The thresholded matrix.

    Returns:
        np.ndarray: The distance matrix.
    """
    zerolongmat_th_dis = 1 - thresholded_matrix  # 1 - similarity to get distance
    distance_matrix = np.zeros_like(zerolongmat_th_dis)

    for s in range(zerolongmat_th_dis.shape[1]):  # Iterate over subjects (columns)
        tempsub = squareform(zerolongmat_th_dis[:, s])  # Convert to squareform
        tempsub[tempsub == 1] = np.inf  # Replace 1 (self-similarity) with np.inf
        distance_matrix[:, s] = squareform(tempsub)  # Convert back to array
    print("Computed the distance matrix for all subjects.")
    return distance_matrix

def save_matrices(thresholded_matrix, distance_matrix, output_directory, output_file_name_prefix):
    """
    Save the thresholded and distance matrices as .npy files.

    Parameters:
        thresholded_matrix (np.ndarray): The thresholded matrix.
        distance_matrix (np.ndarray): The distance matrix.
        output_directory (str): Directory to save the output files.
    """
    os.makedirs(output_directory, exist_ok=True)
    np.save(os.path.join(output_directory, f"thresholded_matrix.npy"), thresholded_matrix)
    np.save(os.path.join(output_directory, f"distance_matrix.npy"), distance_matrix)
    print(f"Matrices saved to {output_directory}")

def preprocess_data(input_directory, input_file_name, percentile, output_directory):
    """
    Full preprocessing pipeline: load data, process, and save output.

    Parameters:
        input_directory (str): Path to the input directory.
        input_file_name (str): Name of the matrix input file (CSV).
        percentile (float): Percentile threshold for thresholding.
        output_directory (str): Directory to save the output matrices.
    """
    print("Starting preprocessing...")
    matrix = load_csv(input_directory, input_file_name)
    r_matrix = convert_to_r_correlation(matrix)
    thresholded_matrix = threshold_matrix(r_matrix, percentile)
    distance_matrix = compute_distance_matrix(thresholded_matrix)
    save_matrices(thresholded_matrix, distance_matrix, output_directory)
    print("Preprocessing completed successfully.")

def main():
    """
    Main function to handle argument parsing and call the preprocess_data function.
    """
    if len(sys.argv) != 5:
        print("Usage: python preprocess_data.py <input_directory> <input_file_name> <percentile_threshold> <output_directory>")
        sys.exit(1)

    input_directory = sys.argv[1]  # path to the input directory
    input_file_name = sys.argv[2]  # name of the input matrix file of Fisher Z correlation as .csv file, where rows are pairwise correlations of the 419 ROIs and columns are participants
    percentile_threshold = float(sys.argv[3])  # Percentile threshold (e.g. 90, where top 10% of connections (positive) are used)
    output_directory = sys.argv[4]  # Output directory to where the two matrices (distance and thresholded) will be saved

    preprocess_data(input_directory, input_file_name, percentile_threshold, output_directory)

if __name__ == "__main__":
    main()

    
    
