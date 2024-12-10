import os
import numpy as np
import sys

def calculate_slope(metric_data_path, output_directory):
    """
    Calculate the slope of LCC between adjacent values.

    Parameters
        metric_data_path: where the s_lcc_org matrix is located
        s_lcc_org (ndarray): The LCC data matrix (participants x nodal iterations).

    Returns:
        minslopeit_LCC (ndarray): The index of the point of the minimum slope for each participant.
    """
    
    s_lcc_org = np.load(os.path.join(metric_data_path, 's_lcc_org.npy'))
    
    # Initialize the output array for the minimum slope indices
    LCCslope = np.full((s_lcc_org.shape[0], s_lcc_org.shape[1] - 1), np.nan)

    # Calculate slopes between adjacent time points for each subject
    for s in range(s_lcc_org.shape[0]):  # Iterate over subjects
        temp = s_lcc_org[s, :]
        for i in range(s_lcc_org.shape[1] - 1):  # Iterate over time points
            x1, x2 = i, i + 1
            y1, y2 = temp[i], temp[i + 1]
            LCCslope[s, i] = (y2 - y1) / (x2 - x1)  # Calculate slope

    # Find the minimum slope for each subject and the corresponding time point
    minslopeit_LCC = np.full(s_lcc_org.shape[0], np.nan)

    for s in range(s_lcc_org.shape[0]): 
        temp = np.nanmin(LCCslope[s, :])  # Find what the minimum slope is
        temp_index = np.where(LCCslope[s, :] == temp)[0]  # Find the index for the point of the minimum slope
        if len(temp_index) > 0:
            minslopeit_LCC[s] = temp_index[0]  # Store the index of the minimum slope

    return minslopeit_LCC

    # Save the results
    os.makedirs(output_directory, exist_ok=True)
    np.save(os.path.join(output_directory, "minslopeit_LCC.npy"), minslopeit_LCC)
    print("Minimum slope indices calculated and saved.")

def main():
    
    if len(sys.argv) != 3:
        print("Usage: python calculate_slope.py <metric_data_path> <output_directory>")
        sys.exit(1)

    preprocessed_data_path = sys.argv[1] #path to thresholded matrix (in output folder)
    output_directory = sys.argv[2] #output directory path

    calculate_slope(metric_data_path, output_directory)

if __name__ == "__main__":
    main()