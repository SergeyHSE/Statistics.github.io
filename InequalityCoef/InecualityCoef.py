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
