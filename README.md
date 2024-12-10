# Targeted_Attack_CR analysis of resting-BOLD connectivity (Argiris, Stern, & Habeck, 2024)

This analysis is a targeted attack analysis of resting-BOLD connectivity matrices, where an individual's connectivity matrix is virtually "lesioned" in-silico over multiple iterations, based on nodal strength. The main outcome measure is the iteration of the greatest slope decline of the largest connected component (LCC), which is tested as a measure of brain resilience and cognitive reserve (CR). Additionally, we calculate a small-world network metric based on the ratio of the global clustering coefficient to the characteristic path length, after each have been normalized to their respective metrics calculated on randomized matrices. 

# Requirements
This project requires the following Python packages to run. All dependencies are listed in the requirements.txt file. To install them, use the following command:

pip install -r requirements.txt

### Key Dependencies:

- `numpy` For numerical computations.  
- `scipy` For scientific computing and advanced mathematical functions.  
- `brainconn` For brain connectivity analysis and graph theory computations.  
- `pandas` For data manipulation and analysis.  
- `statsmodels` (for statistical modeling and linear regressions)  
- `scikit-learn` (for linear regression computations)   

Ensure you have Python (>=3.8) installed before proceeding.  

# Scripts

You will find five scripts to execute. The script (preprocess_data.py) prepares the data matrices that you will find in the "data" folder. The scripts are based on calculation of the LCC on the follow-up (T2) data. However, for statistical analysis where the LCC at follow-up (T2), we also consider the LCC at baseline (T1). You will have to modify the script to process both of these datasets to include in the model.
The data provided is only mock data of 10 participants with 419 x 419 edge connections (87571 edges in total). The matrices are arranged as vectors to comprise a 2D edge x participant matrix. After preprocessing, the compute_metrics_org.py script extracts the graph theory metrics for the original matrices. The compute_metrics_rand.py scripts computes the same metrics (apart from tracking the node removed on each iteration and the nodal degree for all nodes for each iteration, which were done only for the original data matrices), on the randomized data matrices. After, calculate_slope.py calculates the iteration of largest slope decline for the LCC and calculate_small_world.py calculates the small-world statistic across each lesioning iteration.
The statistical_analysis.py script provides the regressions used for the analyses. The small-world calculation for the unlesioned matrix was the only value used from the small-world matrix.The participant data information, at both baseline (T1) and follow-up (T2) should be loaded and are found in the data folder. 

### Computing time and output

In the participant for-loop, the amount of time that it takes for each iteration is listed. On an Apple M1 Chip, the processing time per participant was around 400 seconds (~6.67 minutes). Each matrix is saved as separate output and should go to a specified output directory. At the end of each script, the output is indicated. 

# Graph theory metrics

The analysis was originally performed in MATLAB using brain connectivity toolbox (Sporns & Rubinov, 2010). In the spirit of free software, I tried to adapt it to Python. As I am not a coder, one should certainly check through each of the scripts.

We are mainly working with undirected, weighted graphs, and this is
usually reflected in the function that you will choose to extract each
graph theory metric (e.g., BCT function extensions usually say "wu"
weighted undirected) or "und" (undirected)

- s_lam: lambda (characteristic path length)- The shortest path length between two nodes in a network 
    % %is the minimum number of edges (or connections) that need to be traversed to get from one node to the other. 
    
- s_geff: global efficiency- efficiency of information transfer across the entire network. 
    %  It is  the average of the inverse shortest path length between all pairs of nodes.

- s_clus: clustering coefficient (global)- the likelihood that neighbors of a given node are connected to each other. 
    % It quantifies the tendency of nodes in a network to form clusters, and we take the average across all nodes. 

- s_lcc: largest connected component (primary measure)- it is the largest subgraph, or number of nodes,  
    % in which any pair of nodes is connected directly or indirectly through a series of edges.

- s_mod: modularity- it's how well a network can be separated into clusters where connections within the same cluster 
    % are stronger or more frequent than connections between different clusters. Measure of segregation.

- nodecount: the node that is removed at each threshold
    
- nodedegree: the number of edges connected to each node, at each thresholding level

