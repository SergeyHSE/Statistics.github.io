import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
pd.set_option('display.max_columns', True)

df_houseprice = pd.read_csv('https://raw.githubusercontent.com/bdemeshev/em301/master/datasets/flats_moscow.txt', sep='\s+')
df_houseprice.head()

df_houseprice = df_houseprice.drop('n', axis=1)
df_houseprice
