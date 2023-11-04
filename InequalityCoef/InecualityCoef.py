import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

path = r"C:\Users\User\Documents\книги\ВШЭ\учёба\Статистика\Семинар 5\data (6).xls"
path = path.replace('\\', '/')
data = pd.read_excel(path)
data.head()
data = data.drop('Unnamed: 1', axis=1)
data.head()

data['Регион'] = data['Регион'].str.strip()
data
data.to_excel('DataGini.xlsx')
pd.set_option('display.max_rows', None)
data

def lorenz_curve_gini(data, columnName=None):
    if isinstance(data, (list, np.ndarray)):
        data = np.array(data)
        sorted_data = np.sort(data)
    elif isinstance(data, pd.DataFrame):
        if columnName is not None:
            sorted_data = data[columnName].sort_values(ascending=True).values
        else:
            raise ValueError("If data is a DataFrame, columnName must be specified.")
    else:
        raise ValueError("data should be a DataFrame, list, or NumPy array.")
    
    total_income = sorted_data.sum()
