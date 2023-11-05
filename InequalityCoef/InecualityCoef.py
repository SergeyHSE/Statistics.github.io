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

gini_2018 = lorenz_curve_gini(data, '2018')
print(f'Gini coef for 2018: {gini_2018}')

gini_2020 = lorenz_curve_gini(data, '2020')
print(f'Gini coef for 2020: {gini_2020}')

gini_2022 = lorenz_curve_gini(data, '2022')
print(f'Gini coef for 2022: {gini_2022}')

"""
Interval method
"""

def interval_lorenz_curve_gini(data, columnName=None):
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
    
    n = len(sorted_data)
    custom_percent_rank = np.arange(1, n + 1) / n
    result_list, _ = np.histogram(custom_percent_rank, bins=10)
    share_popul = result_list / n
    cumulative_share = np.cumsum(share_popul)
    sums = []
    start_ind = 0 
    for count in result_list:
        end_ind = start_ind + count
        subset = np.array(sorted_data)[start_ind: end_ind]
        decile_sum = np.sum(subset)
        sums.append(decile_sum)
        start_ind = end_ind
    relative_values = share_popul * sums
    cumulative_values = np.cumsum((relative_values / sum(relative_values)))
    palma_ratio = (cumulative_values[-1] - cumulative_values[-2]) / cumulative_values[3]        
    new_array = [cumulative_values[0]]
    for i in range(1, len(cumulative_values)):
        new_array.append(cumulative_values[i] + cumulative_values[i - 1])
        
