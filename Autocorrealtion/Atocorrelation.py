import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.stattools as adfuller
pd.set_option('display.max_columns', None)


path = r".........\Кисломолоная_продукция.xlsx"
path = path.replace('\\', '/')
df = pd.read_excel(path, sheet_name='data')

# Check NaN
df.isna().any()

df.head()

df = df.drop('Удаление автокорреляци', axis=1)

# Fill NaN
df['Year'].fillna(method='ffill', inplace=True)
df
