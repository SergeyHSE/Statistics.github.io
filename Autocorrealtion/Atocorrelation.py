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
