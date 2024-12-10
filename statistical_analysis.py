import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# File paths
input_dir = 'input_directory/'
output_dir = 'output_directory/'

# Load data from input directory
database = pd.read_csv(input_dir + 'data_baseline.csv')
datalong = pd.read_excel(input_dir + 'data_longitudinal.xlsx')

# Load the minslope_LCC and small-world network data from the output directory
minslopeit_LCC = np.loadtxt(output_dir + 'minslopeit_LCC.txt')  # Replace with actual file name and format
s_clus_org = pd.read_csv(output_dir + 's_clus_org.csv')  # Replace with actual file format if different
s_clus_rand = pd.read_csv(output_dir + 's_clus_rand.csv')  # Replace with actual file format if different
s_lam_org = pd.read_csv(output_dir + 's_lam_org.csv')  # Replace with actual file format if different
s_lam_rand = pd.read_csv(output_dir + 's_lam_rand.csv')  # Replace with actual file format if different

# Assuming the following are column names in your data files (adjust as needed)
CT = 'CT_column_name'  # Replace with the actual column name for CT
NP_Reason_Z = 'NP_Reason_Z_column_name'  # Replace with the actual column name for NP_Reason_Z
Age = 'Age_column_name'  # Replace with the actual column name for Age
Edu = 'Edu_column_name'  # Replace with the actual column name for Edu
Sex = 'Sex_column_name'  # Replace with the actual column name for Sex
NART = 'NART_column_name'  # Replace with the actual column name for NART

# Assuming minslopeit_LCC is available as a variable
# Load or define the minslopeit_LCC, CTlong, longscrub, s_lcc_org, and SD_sw_l1 as needed

# CT residualized
CT_diff = datalong[CT] - database[CT]
X = sm.add_constant(database[CT])
model = sm.OLS(CT_diff, X).fit()
CT_diffres = model.resid

# Behavior residualized
fluid_diff = datalong[NP_Reason_Z] - database[NP_Reason_Z]

# minslopeit_LCC standardized for interaction term
minslopeit_LCC_s = (minslopeit_LCC - np.nanmean(minslopeit_LCC)) / np.nanstd(minslopeit_LCC)

# Regression for small-world network with slope drop
SD_sw_long = []
for s in range(s_clus_org.shape[0]):
    for i in range(s_clus_org.shape[1]):
        SD_sw_long.append((s_clus_org.iloc[s, i] / s_clus_rand.iloc[s, i]) / (s_lam_org.iloc[s, i] / s_lam_rand.iloc[s, i]))
SD_sw_long = np.array(SD_sw_long).reshape(s_clus_org.shape)  # Reshape to match the expected dimensions

# Regression for small-world network analysis
X1 = pd.DataFrame({
    'Age': database[Age],
    'Edu': database[Edu],
    'Sex': database[Sex],
    'NART': database[NART],
    'CTlong': CTlong,  # Define CTlong variable
    'longscrub': longscrub,  # Define longscrub variable
    'minslopeit_LCC_base': minslopeit_LCC_base,  # Define minslopeit_LCC_base variable
    's_lcc_org_1': s_lcc_org.iloc[:, 0],  # Define s_lcc_org
    'SD_sw_l1': SD_sw_l1  # Define SD_sw_l1 variable
})
X1 = sm.add_constant(X1)
y1 = minslopeit_LCC
model1 = sm.OLS(y1, X1).fit()
print(model1.summary())

# Behavior regression with interaction
X2 = pd.DataFrame({
    'Age': database[Age],
    'Edu': database[Edu],
    'Sex': database[Sex],
    'NART': database[NART],
    'CT_diffres': CT_diffres,
    'minslopeit_LCC_s': minslopeit_LCC_s,
    'longscrub': longscrub,
    's_lcc_org_1': s_lcc_org.iloc[:, 0],
    'NP_Reason_Z': database[NP_Reason_Z]
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
p_sign = p_values * np.sign(t_stats)
print(p_sign)
