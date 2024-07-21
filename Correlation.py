import pandas as pd

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Compute the correlation coefficients
correlation = data.corr()['Outcome'].abs().sort_values(ascending=False)

# Print the correlation coefficients
print(correlation)
