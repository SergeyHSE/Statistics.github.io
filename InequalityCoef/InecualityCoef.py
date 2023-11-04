import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

path = r"............\data (6).xls"
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
    cumulative_income = np.cumsum(sorted_data) / total_income
    cumulative_population = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    plt.figure(figsize=(8, 8), dpi=90)
    plt.plot(cumulative_population, cumulative_income, label='Lorenz curve',
             color='tab:red', linewidth=2.0)
    plt.plot([0, 1], [0, 1], 'k--', label='Equality Line',
             linewidth=2.0, color='tab:green')
    plt.xlabel('Cumulative % of Population', fontsize=14)
    plt.ylabel(f'Cumulative % of {columnName} Income', fontsize=14)
    plt.title(f'Lorenz Curve for {columnName}', fontsize=18)
    plt.grid()
    plt.gca().set_aspect('equal')
    gini_coef = 2 * (0.5 - np.trapz(cumulative_income, cumulative_population))
    plt.text(0.6, 0.25, f'Gini coef: {gini_coef:.3f}', fontsize=16, color='darkblue')
    plt.legend()
    plt.show()


    return gini_coef

"""
Interval method
"""

