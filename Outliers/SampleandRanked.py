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
