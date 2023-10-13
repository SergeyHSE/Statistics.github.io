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
