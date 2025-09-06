import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
df = pd.read_csv(r"/home/comp/Downloads/german_credit.csv")

print(df.head())
results = []

Target = 'Creditability'
Categorical_Cols = ['Purpose', 'Sex & Marital Status', 'Guarantors',
 'Most valuable available asset', 'Type of apartment',
 'Occupation', 'Telephone', 'Foreign Worker', 'Account Balance',
 'Payment Status of Previous Credit', 'Value Savings/Stocks',
 'Length of current employment', 'Instalment per cent',
 'Duration in Current address', 'Concurrent Credits',
 'No of Credits at this Bank', 'No of dependents']

for col in Categorical_Cols:
    contingency_table = pd.crosstab(df[col], df[Target])
    chi2, p, dof, ex = chi2_contingency(contingency_table)
    results.append({'Feature': col, 'Chi2': chi2, 'p-value': p})

results_df = pd.DataFrame(results).sort_values('p-value')

print(results_df)

# Results:
#                               Feature        Chi2       p-value
# 8                     Account Balance  123.720944  1.218902e-26
# 9   Payment Status of Previous Credit   61.691397  1.279187e-12
# 10               Value Savings/Stocks   36.098928  2.761214e-07
# 3       Most valuable available asset   23.719551  2.858442e-05
# 4                   Type of apartment   18.674005  8.810311e-05
# 0                             Purpose   33.356447  1.157491e-04
# 11       Length of current employment   18.368274  1.045452e-03
# 14                 Concurrent Credits   12.839188  1.629318e-03
# 7                      Foreign Worker    5.821576  1.583075e-02
# 1                Sex & Marital Status    9.605214  2.223801e-02
# 2                          Guarantors    6.645367  3.605595e-02


# 12                Instalment per cent    5.476792  1.400333e-01
# 6                           Telephone    1.172559  2.788762e-01
# 15         No of Credits at this Bank    2.671198  4.451441e-01
# 5                          Occupation    1.885156  5.965816e-01
# 13        Duration in Current address    0.749296  8.615521e-01
# 16                   No of dependents    0.000000  1.000000e+00

target_col = 'Creditability'

continuous_vars = ['Age (years)', 'Duration of Credit (month)', 'Credit Amount']

for var in continuous_vars:
    group0 = df[df[target_col] == 0][var]
    group1 = df[df[target_col] == 1][var]

    f_stat, p_val = f_oneway(group0, group1)

    print(f"ANOVA result for {var}: F-statistic = {f_stat:.3f}, p-value = {p_val:.3e}")

# ANOVA result for Age (years): F-statistic = 8.384, p-value = 3.868e-03
# ANOVA result for Duration of Credit (month): F-statistic = 48.334, p-value = 6.488e-12
# ANOVA result for Credit Amount: F-statistic = 24.483, p-value = 8.795e-07
