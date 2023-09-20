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
