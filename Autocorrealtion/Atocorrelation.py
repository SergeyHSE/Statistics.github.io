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

month_map = {
    'январь': 'January',
    'февраль': 'February',
    'март': 'March',
    'апрель': 'April',
    'май': 'May',
    'июнь': 'June',
    'июль': 'July',
    'август': 'August',
    'сентябрь' : 'September',
    'октябрь' : 'October',
    'ноябрь' : 'November',
    'декабрь' : 'December'
}

df['Month'] = df['Month'].map(month_map)
df
df['Date'] = pd.to_datetime(df['Year'].astype(int).astype(str) + df['Month'], format='%Y%B')

df = df.drop(['Year', 'Month'], axis=1)
df.columns
df

# Create dummy variables
df['Q2'] = 0
df['Q3'] = 0
df['Q4'] = 0

def get_quarter(date):
    if date.month in [3, 4, 5]:
        return 'Q2'
    elif date.month in [6, 7, 8]:
        return 'Q3'
    elif date.month in [9, 10, 11]:
        return 'Q4'
    else:
        return 'Q1'
df

df['Quarter'] = df['Date'].apply(get_quarter)
df = pd.get_dummies(df, columns=['Quarter'], drop_first=True)
df = df.drop(['Q2', 'Q3', 'Q4'], axis=1)
df.columns

df['Q2'] = df['Quarter_Q2']
df['Q3'] = df['Quarter_Q3']
df['Q4'] = df['Quarter_Q4']
df = df.drop(['Quarter_Q2', 'Quarter_Q3', 'Quarter_Q4'], axis=1)

# Decompose data by selecting the appropiate frequancy
decomp = sm.tsa.seasonal_decompose(
    df['Production'], period=12)

plt.figure(figsize=(10, 8), dpi=100)
decomp_plot = decomp.plot()
plt.title('Production dynamics')
decomp_plot .set_figheight(10)
decomp_plot .set_figwidth(10)
plt.show()

# Let's fit regression model and check all parametrs
X = df[['price_opt', 'RMCI', 'Q2', 'Q3', 'Q4']]
y = df['Production']

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print(model.summary())

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:             Production   R-squared:                       0.468
Model:                            OLS   Adj. R-squared:                  0.380
Method:                 Least Squares   F-statistic:                     5.287
Date:                Sat, 14 Oct 2023   Prob (F-statistic):            0.00134
Time:                        11:57:20   Log-Likelihood:                 119.61
No. Observations:                  36   AIC:                            -227.2
Df Residuals:                      30   BIC:                            -217.7
Df Model:                           5                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          0.2849      0.072      3.938      0.000       0.137       0.433
price_opt      0.0006      0.002      0.361      0.721      -0.003       0.004
RMCI          -0.0008      0.001     -1.164      0.254      -0.002       0.001
Q2             0.0155      0.005      3.358      0.002       0.006       0.025
Q3             0.0143      0.005      2.825      0.008       0.004       0.025
Q4            -0.0015      0.005     -0.300      0.766      -0.012       0.009
==============================================================================
Omnibus:                        1.661   Durbin-Watson:                   2.032
Prob(Omnibus):                  0.436   Jarque-Bera (JB):                1.310
Skew:                           0.268   Prob(JB):                        0.519
Kurtosis:                       2.234   Cond. No.                     4.73e+03
==============================================================================
"""

# Compare target and ather variables
fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
axes[0].plot(df['Date'], df['Production'], label='Production', linestyle='-', marker='o')
axes[0].set_title('Production time series')
axes[0].set_ylabel('Production')
axes[0].grid(True)
axes[1].plot(df['Date'], df['price_opt'], label='Price', linestyle='-', marker='o', color='orange')
axes[1].set_title('Price time series')
axes[1].set_ylabel('Price')
axes[1].grid(True)
axes[2].plot(df['Date'], df['RMCI'], label='RMCI', linestyle='-', marker='o', color='green')
axes[2].set_title('RMCI time series')
axes[2].set_xlabel('date')
axes[2].set_ylabel('RMCI')
axes[2].grid(True)
