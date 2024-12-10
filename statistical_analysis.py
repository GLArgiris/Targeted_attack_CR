import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


'minslopeit_LCC_base': minslopeit_LCC_base,  

# File paths
input_dir = 'input_directory/' # manually indicate the input directory
output_dir = 'output_directory/' # manually indicate the input directory

# Load data from input directory
database = pd.read_csv(input_dir + 'data_baseline.csv')
datalong = pd.read_csv(input_dir + 'data_longitudinal.csv')

# Load the minslope_LCC and small-world network data from the output directory
minslopeit_LCC = np.load(os.path.join(output_dir, 'minslopeit_LCC.npy'))  # Iteration of slope decay of LCC
SD_sw_long = np.load(os.path.join(output_dir, 'SD_sw_long.npy'))

# Define variables based on the columns from the data files

# CT residualized
CT_diff = datalong[CT] - database[CT]
X = sm.add_constant(database[CT])
model = sm.OLS(CT_diff, X).fit()
CT_diffres = model.resid


# Behavior residualized
fluid_diff = datalong['NP_Reason_Z'] - database['NP_Reason_Z']

# minslopeit_LCC standardized for interaction term
minslopeit_LCC_s = (minslopeit_LCC - np.nanmean(minslopeit_LCC)) / np.nanstd(minslopeit_LCC)


# Regression for longitudinal minslopeit_LCC with small-world property included. 
# NOTE! For completion, minslopeit_LCC should also be calculated at baseline to adjust for this in the model. The model does not contain this term at the moment.  
X1 = pd.DataFrame({
    'Age': database['Age'],
    'Edu': database['Edu'],
    'Sex': database['Sex'],
    'NART': database['NART'],
    'CTlong': datalong['CT'],  # Define CTlong variable (longitudinal CT)
    'longscrub': datalong['Scrub'],  # Define longscrub variable
    's_lcc_org_1': s_lcc_org[:, 0],  # Define s_lcc_org
    'SD_sw_l1': SD_sw_long[:, 0]  # Tale the small-world value of the unlesioned matrix
})
X1 = sm.add_constant(X1)
y1 = minslopeit_LCC
model1 = sm.OLS(y1, X1).fit()
print(model1.summary())

# Behavior regression with interaction
X2 = pd.DataFrame({
    'Age': database['Age'],
    'Edu': database['Edu'],
    'Sex': database['Sex'],
    'NART': database['NART'],
    'CT_diffres': CT_diffres,
    'minslopeit_LCC_s': minslopeit_LCC_s,
    'longscrub': datalong['Scrub'],
    's_lcc_org_1': s_lcc_org[:, 0],
    'NP_Reason_Z': database['NP_Reason_Z']
})
X2 = sm.add_constant(X2)
y2 = fluid_diff
model2 = sm.OLS(y2, X2).fit()

# Add interaction term
X2['interaction'] = X2['minslopeit_LCC_s'] * X2['NP_Reason_Z']
model2_with_interaction = sm.OLS(y2, X2).fit()

print(model2_with_interaction.summary())

# Extract p-values with sign of t-statistics for the coefficients
p_values = model2_with_interaction.pvalues[1:].values  # Exclude constant term
t_stats = model2_with_interaction.tvalues[1:].values  # Exclude constant term
p_sign = p_values * np.sign(t_stats) #I usually look at the signed p-value
print(p_sign)
