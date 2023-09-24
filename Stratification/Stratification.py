# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 22:39:56 2023

@author: SergeyHSE

We are gonna extract colums kike region, revenue and pig farming cost,
sort them by revenue, remove zeros. We need to find intervals that help us split our data on
5 parts (stratums or cohorts). 
Then we need to find mean value for cohorts and population, calculate variation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
pd.set_oprion('display.max_columns', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%3f' %x)

# Firstly let's load data and extract nesessary columns
# Because our columns are lacated in different sheets, we need combine them

path = r"Your path\СХО_данные для группировки.xlsx"
path = path.replace('\\', '/')
data_sheet1 = pd.read_excel(path, sheet_name='1')
data_sheet2 = pd.read_excel(path, sheet_name='2')
data_sheet1 = data_sheet1[['Субъект РФ', '10050']]
data_sheet1.head(10)
data_sheet2 = data_sheet2['14070']
data_sheet2.head(10)
data_sheet1.shape
data_sheet2.shape
data = pd.concat([data_sheet1, data_sheet2], axis=1)
data.head(10)

# Remove zeros

(data == 0).sum()
data = data[(data != 0).all(axis=1)]
data.shape

# Find quantiles and IQR

data = data.sort_values(by=['10050'])
count = data.shape[0]
count
quantile1 = np.quantile(data['10050'], 0.25)
quantile1
quantile3 = np.quantile(data['10050'], 0.75)
quantile3
