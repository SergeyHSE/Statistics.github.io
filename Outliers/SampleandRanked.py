Created on Sat Sep 16 17:52:43 2023

@author: SergeyHSE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
pd.set_option('display.max_columns', None)

path = r"Your path.xlsx"
path = path.replace('\\', '/')

data = pd.read_excel(path, sheet_name='2')
data.head()
data_costs = data['14070']
data_costs.head()
data_costs.shape
data_costs.isnull().any().any()
(data_costs == 0).sum() #Calculate number of zeros

# Delete all rows with zeros value

data_costs = data_costs.loc[(data_costs != 0)]
data_costs.shape
data_costs.head(n=20)

# Build hist

plt.figure(figsize=(10, 8), dpi=100)
plt.hist(data_costs, bins=50, color='tab:green')
plt.title('Distribution of pig farming costs')
plt.xlabel('Costs')
plt.ylabel('Count')
plt.show()

# Build outliers box
plt.figure(figsize=(6, 6), dpi=100)
plt.boxplot(data_costs)
plt.title('Outliers of pig farming costs', fontsize=20)
plt.show()

# Apply inter quartile range (IQR) score method

q1 = np.quantile(data_costs, 0.25)
q3 = np.quantile(data_costs, 0.75)
med = np.median(data_costs)
iqr = q3-q1
upper_bound = q3+(1.5*iqr) 
lower_bound = q1-(1.5*iqr)
print(iqr, upper_bound, lower_bound)

outliers = data_costs[(data_costs <= lower_bound) | (data_costs >= upper_bound)]
print('Outliers:{}'.format(outliers))

data_without_outliers = data_costs[(data_costs <= lower_bound) | (data_costs >= upper_bound)]

data_without_outliers.shape

plt.figure(figsize=(6, 6), dpi=100)
plt.boxplot(data_without_outliers)
plt.title('Pig farming costs without outliers (IQR)', fontsize=20)
plt.show()

plt.figure(figsize=(10, 8), dpi=100)
plt.hist(data_without_outliers, bins=50, color='tab:green')
plt.title('Distribution of pig farming costs (Without outliers)')
plt.xlabel('Costs')
plt.ylabel('Count')
plt.show()

z = np.abs(stats.zscore(data_costs)) 
     
data_Z_score = data_costs[(z<3)]
data_Z_score.shape
data_costs.shape
data_without_outliers.shape

plt.figure(figsize=(6, 6), dpi=100)
plt.boxplot(data_Z_score)
plt.title('Pig farming costs without outliers (Z_score)', fontsize=20)
plt.show()

# Get description

description = data_Z_score.describe()
description.to_csv('Description.csv')
print(description)

# Find the directory
directory = os.getcwd()
filename = 'Description.csv'
file_path = os.path.join(directory, filename)
if os.path.exists(file_path):
    print(f"The file '{filename}' is located at: {file_path}")
else:
    print(f"The file '{filename}' was not found in the current directory")
outlier_result_table = pd.DataFrame([[data_costs.shape[0], data_Z_score.shape[0], data_without_outliers.shape[0]]],
                                    columns=['Initial data', 'Z-score', 'IQR'])
outlier_result_table.to_csv('Results_of_outlier_cut.csv')

# quantile method (you shouldn't try this method, because it shows only minimum or maximum of sample)
quantile = data_Z_score.quantile(0.1)
data_ranked = data_Z_score[data_Z_score < quantile]
data_ranked.shape
data_ranked

# Method of extracting every 6th values from every 10 of sample
data_Z_score.head()
data_sort = data_Z_score.sort_values()
data_sort.head(20)
data_sort.shape
total_rows = len(data_sort)
total_rows
selected_values = []
for i in range(0, total_rows, 10):
    if i + 5 < total_rows:
        selected_values.append(data_sort.iloc[i + 5])

len(selected_values)
data_ranked = pd.DataFrame({'SelectedValues': selected_values})
data_ranked.shape
data_ranked = data_ranked.squeeze()
data_ranked.shape

random_description = data_random.describe()
random_description.to_csv('Random_description.csv')
ranked_description = data_ranked.describe()
ranked_description.to_csv('Ranked_description.csv')
