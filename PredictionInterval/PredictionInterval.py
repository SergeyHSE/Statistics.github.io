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

X = df_houseprice.drop('price', axis=1)
y = df_houseprice['price']
X.shape, y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train).fit()

X_test = sm.add_constant(X_test)
y_pred = model.predict(X_test)
r_2 = r2_score(y_test, y_pred)
print(f'R2: {r_2}')

