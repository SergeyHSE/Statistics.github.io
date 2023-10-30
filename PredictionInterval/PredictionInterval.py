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

print(model.summary())
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  price   R-squared:                       0.689
Model:                            OLS   Adj. R-squared:                  0.687
Method:                 Least Squares   F-statistic:                     399.3
Date:                Sat, 07 Oct 2023   Prob (F-statistic):               0.00
Time:                        12:27:21   Log-Likelihood:                -7750.8
No. Observations:                1632   AIC:                         1.552e+04
Df Residuals:                    1622   BIC:                         1.558e+04
Df Model:                           9                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const        -13.1351      6.157     -2.134      0.033     -25.211      -1.060
totsp          1.5614      0.127     12.258      0.000       1.312       1.811
livesp         1.0311      0.194      5.306      0.000       0.650       1.412
kitsp          2.2895      0.432      5.295      0.000       1.441       3.138
dist          -3.0757      0.232    -13.270      0.000      -3.530      -2.621
metrdist      -1.3188      0.181     -7.283      0.000      -1.674      -0.964
walk           8.5869      1.530      5.613      0.000       5.586      11.588
brick          8.7988      1.688      5.212      0.000       5.487      12.110
floor          6.9686      1.728      4.032      0.000       3.579      10.358
code          -2.6287      0.326     -8.069      0.000      -3.268      -1.990
==============================================================================
Omnibus:                      818.784   Durbin-Watson:                   2.018
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            11133.216
Skew:                           2.007   Prob(JB):                         0.00
Kurtosis:                      15.150   Cond. No.                         796.
==============================================================================
"""
