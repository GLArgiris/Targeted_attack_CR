import numpy as np

def calculate_slope(s_lcc_org):
    """
    Calculate the slope of LCC between adjacent values.

    Args:
        s_lcc_org (ndarray): The LCC data matrix (participants x nodal iterations).

    Returns:
        minslopeit_LCC (ndarray): The index of the point of the minimum slope for each participant.
    """
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

def main():
    """
    Main function to load data and compute slopes.
    """
    # Load s_lcc_org from the file (assuming it's saved in a .npy file)
    s_lcc_org = np.load("s_lcc_org.npy")  # Replace with the correct path to the matrix in the output folder

    # Calculate minimum slope indices
    minslopeit_LCC = calculate_slope(s_lcc_org)

    # Save only the minslopeit_LCC array to a file
    np.save("minslopeit_LCC.npy", minslopeit_LCC)
    print("Minimum slope indices calculated and saved.")

if __name__ == "__main__":
    main()
