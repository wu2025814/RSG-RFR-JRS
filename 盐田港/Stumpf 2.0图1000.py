import numpy as np
import pandas as pd
from scipy.stats import linregress

# Read data from CSV with dtype handling
image_data = pd.read_csv('train_set3000.csv', low_memory=False)

# Ensure numeric types for the necessary columns
image_data['Band_1'] = pd.to_numeric(image_data['Band_1'], errors='coerce')
image_data['Band_2'] = pd.to_numeric(image_data['Band_2'], errors='coerce')
image_data['depth'] = pd.to_numeric(image_data['depth'], errors='coerce')

# Drop rows with NaN values if any after conversion
image_data.dropna(subset=['Band_1', 'Band_2', 'depth'], inplace=True)

# Extract r2, r3 columns, and depth
r2_data = image_data['Band_1'].values
r3_data = image_data['Band_2'].values
dep_data = image_data['depth'].values

# Stumpf method for SDB (Blue/Green)
# Check for valid data before logarithm


a = np.log(1000 * np.pi * np.abs(r2_data))  # blue
b = np.log(1000 * np.pi * np.abs(r3_data))  # green

# Calculate Cbg, replacing division by zero safely
c = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
Cbg = c

# Clear NaN values
Cbg = np.where(np.isnan(Cbg), np.nan, Cbg)

# Build feature matrix
C11 = np.column_stack((np.ones_like(Cbg), Cbg))

# Reshape the target vector
dep_data_column = dep_data.reshape(-1, 1)

# Linear regression fit
reg = linregress(C11[:, 1], dep_data_column.flatten())

# Extract regression coefficients
a0 = reg.intercept
a1 = reg.slope

# Compute predicted values
ZSbg = a0 + a1 * Cbg

# Calculate correlation
R = np.corrcoef(dep_data, Cbg, rowvar=False)[0, 1]
RR2 = R**2 if R is not np.nan else np.nan  # Guard against NaN

print('R2 (blue-green) =', RR2)
print('SDB_Stumpf(bg) =', round(a1, 2), ' x pSDB_bg +', round(a0, 2))

# Read new data with dtype handling
new_data = pd.read_csv('pred_set1000.csv', low_memory=False)

# Ensure numeric types for the necessary columns
new_data['Band_1'] = pd.to_numeric(new_data['Band_1'], errors='coerce')
new_data['Band_2'] = pd.to_numeric(new_data['Band_2'], errors='coerce')

# Extract r2 and r3 columns
r2_data = new_data['Band_1'].values
r3_data = new_data['Band_2'].values



a = np.log(1000 * np.pi * np.abs(r2_data))  # blue
b = np.log(1000 * np.pi * np.abs(r3_data))  # green
c = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
Cbg = c

# Build feature matrix for predictions
C11_new = np.column_stack((np.ones_like(Cbg), Cbg))

# Use trained model parameters for predictions
predicted_depth = a0 + a1 * Cbg

# Load existing CSV and remove unnecessary columns
existing_data = pd.read_csv('pred_set1000.csv', low_memory=False)
existing_data = existing_data.drop(columns=['Band_1', 'Band_2', 'Band_3', 'Band_4', 'Band_5', 'Band_6', 'Band_7', 'Band_8', 'Band_10'])

# Add predicted shallow depth data as a new column
existing_data['stumpf'] = predicted_depth

# Save updated DataFrame back to a new CSV file
output_file = 'result1001.csv'  # Change this path if needed
try:
    existing_data.to_csv(output_file, index=False)
except Exception as e:
    print(f"An error occurred while saving the file: {e}")
