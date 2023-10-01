# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 22:39:56 2023

@author: SergeyHSE

We are gonna extract colums like region, revenue and pig farming cost,
sort them by revenue, remove zeros. We need to find intervals that help us split our data on
5 parts (stratums or cohorts). 
Then we need to find mean value for cohorts and population, calculate variation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
pd.set_oprion('display.max_columns', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%3f' %x)

# Firstly let's load data and extract nesessary columns
# Because our columns are lacated in different sheets, we need combine them

path = r"Your path\СХО_данные для группировки.xlsx"
path = path.replace('\\', '/')
data_sheet1 = pd.read_excel(path, sheet_name='1')
data_sheet2 = pd.read_excel(path, sheet_name='2')
data_sheet1 = data_sheet1[['Субъект РФ', '10050']]
data_sheet1.head(10)
data_sheet2 = data_sheet2['14070']
data_sheet2.head(10)
data_sheet1.shape
data_sheet2.shape
data = pd.concat([data_sheet1, data_sheet2], axis=1)
data.head(10)

# Remove zeros

(data == 0).sum()
data = data[(data != 0).all(axis=1)]
data.shape

# Remove outliers
z_threshold = 3

z_scores = np.abs(stats.zscore(data['14070']))

filtered_data = data[z_scores < z_threshold]

# Find quantiles and IQR

data = data.sort_values(by=['10050'])
count = data.shape[0]
count
quantile1 = np.quantile(data['10050'], 0.25)
quantile1
quantile3 = np.quantile(data['10050'], 0.75)
quantile3

iqr = quantile3 - quantile1
iqr
# Now we are gonna calculate best interval for grouping (h)
h = (2*iqr) / (pow(count, 1/3))
h
max_index = data['10050'].max()
max_index
x = max_index / h
x # this is a maximum number of intervals

# In next code there is formed data to build hist

banch = []
strat_count = 0

while strat_count <= max_index:
    banch.append(strat_count)
    strat_count += h

# Add the last value even if strat_count exceeds max_index
if strat_count - h != max_index:
    banch.append(max_index)

type(banch)
len(banch)
banch

# Create DataFrame of hist
frequency_df = pd.DataFrame(columns=['Banch', 'Frequency'])

for i in range(len(banch) - 1):
    lower_bound = banch[i]
    upper_bound = banch[i + 1]
    count = len(data[(data['10050'] >= lower_bound) & (data['10050'] < upper_bound)])
    frequency_df.loc[i] = [lower_bound, count]

# Add the last bucket
lower_bound = banch[-1]
count = len(data[data['10050'] >= lower_bound])
frequency_df.loc[len(banch) - 1] = [lower_bound, count]
frequency_df

strat_boundaries = [0, 19000, 44000, 84000, 168000, float('inf')]
strata_labeles = ['small_revenue', 'pre_middle_revenue', 'middle_revenue', 'upper_middle_revenue', 'large_revenue']
data['strata'] = pd.cut(data['10050'], bins=strat_boundaries, labels=strata_labeles)

data['strata'].value_counts()
data.head()

# split this date by strats

data_small = data[(data['strata']=='small_revenue')]
data_small.shape
data_small
data_pre_middle = data[(data['strata']=='pre_middle_revenue')]
data_pre_middle.shape
data_middle = data[(data['strata']=='middle_revenue')]
data_middle.shape
data_upper_middle = data[(data['strata']=='upper_middle_revenue')]
data_upper_middle.shape
data_lardge = data[(data['strata']=='large_revenue')]
data_lardge.shape
data_lardge.head()

# Find the average values ​​of the indicator and dispersion according to the general general scheme,
# for each strategy

results_df = pd.DataFrame(columns=['Strata', 'Count', 'Percentage', 'Mean', 'Variance'])

for strata in strata_labeles:
    current_strata_data = data[data['strata'] == strata]
    count = current_strata_data.shape[0]
    percentage = count / data.shape[0]
    mean = current_strata_data['14070'].mean()
    variance = current_strata_data['14070'].var()
    results_df = results_df.append({'Strata': strata, 'Count': count, 'Percentage': percentage,
                                    'Mean': mean, 'Variance': variance}, ignore_index=True)

total_count = data.shape[0]
total_percentage = 1.0
total_mean = data['14070'].mean()
total_var = data['14070'].var()
results_df = results_df.append({'Strata' : 'Population',
                                'Count' : total_count,
                                'Percentage' : total_percentage,
                                'Mean' : total_mean,
                                'Variance' : variance}, ignore_index=True)

print(results_df)
results_df.to_excel('Table_for_all_strats.xlsx')

# We should copy the previous code to return table in necessary shape

for strata in strata_labeles:
    current_strata_data = data[data['strata'] == strata]
    count = current_strata_data.shape[0]
    percentage = count / data.shape[0]
    mean = current_strata_data['14070'].mean()
    variance = current_strata_data['14070'].var()
    results_df = results_df.append({'Strata': strata, 'Count': count, 'Percentage': percentage,
                                    'Mean': mean, 'Variance': variance}, ignore_index=True)

# Calculate real mean

results_df['real_mean'] = results_df['Percentage']*results_df['Mean']
results_df.head()
sum_real_mean = results_df['real_mean'].sum()
sum_real_mean

"""
Calculate sample variance and Deff 
"""

# Before this we should form new samples, each of which contain ten percent of cohorts

def select_rows(df):
    selected_rows = []
    for i in range(5, len(df), 10):
        selected_rows.append(df.iloc[i:i+1])
    return pd.concat(selected_rows, axis=0)

selected_data_small = select_rows(data_small)
selected_data_pre_middle = select_rows(data_pre_middle)
selected_data_middle = select_rows(data_middle)
selected_data_upper_middle = select_rows(data_upper_middle)
selected_data_large = select_rows(data_lardge)
selected_data_large.head()

# Now let's calculate number of row of every new sample

dataframes = [selected_data_small, selected_data_pre_middle, selected_data_middle,
              selected_data_upper_middle, selected_data_large]
row_counts = []
for d in dataframes:
    row_counts.append(len(d))
sum_rows_sample = sum(row_counts)    
sum_rows_sample

# Create new column to calculate sum of cohort variance

results_df['selcetive_variation'] = results_df['Percentage']*results_df['Variance']
sum_select_var = results_df['selcetive_variation'].sum()
sum_select_var

total_count = data.shape[0]

# Create sample variance function and calculate it

def selective_variation(sum_var, sum_rows, total_rows):
    select_var = ((1 - (sum_rows/total_rows))/sum_rows)*sum_var
    return select_var
selective_variance = selective_variation(sum_select_var, sum_rows_sample, total_count)
selective_variance

# Create variation for population
var_popul = data['14070'].var()

# Create and calculate Deff function for cohorts

def DEff(var_pop, sum_rows, total_rows, select_var):
    deff = select_var/(((1-(sum_rows/total_rows))*var_pop)/sum_rows)
    return deff

deff = DEff(var_popul, sum_rows_sample, total_count, selective_variance)
deff

# let's look at diffference between real mean and mean of cohorts

results_df
y_mean = np.array(results_df['Mean'])
y_mean
strats = results_df['Strata'].tolist()
strats
difference = [mean - sum_real_mean for mean in y_mean]
difference

color_palette = plt.cm.viridis(np.linspace(0, 1, len(strats)))

fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
x = np.arange(len(strats))
for i in range(len(strats)):
    ax.bar(x[i], difference[i], color=color_palette[i], label=strats[i])
ax.set_xlabel('Strat')
ax.set_ylabel('Difference')
ax.set_xticks(x)
ax.set_xticklabels(strats, rotation=45, ha='right')
ax.set_title('Difference between Mean and True Mean')
ax.legend(loc="upper left")
plt.show()

"""
 We already create 10% samples and now we should calculate the same metrix
"""

# Firstly we should combine our tables

dataframes_to_concat = [selected_data_small, selected_data_pre_middle, selected_data_middle, selected_data_upper_middle, selected_data_large]

combined_data = pd.concat(dataframes_to_concat, ignore_index=True)
combined_data.head()
combined_data.shape

# Again create result table for indicators

results_df_rand = pd.DataFrame(columns=['Strata', 'Count', 'Percentage', 'Mean', 'Variance'])
for strata in strata_labeles:
    rand_strata_data = combined_data[combined_data['strata'] == strata]
    count = rand_strata_data.shape[0]
    percentage = count / combined_data.shape[0]
    mean = rand_strata_data['14070'].mean()
    variance = rand_strata_data['14070'].var()
    results_df_rand = results_df_rand.append({'Strata': strata, 'Count': count, 'Percentage': percentage,
                                    'Mean': mean, 'Variance': variance}, ignore_index=True)
total_count = data.shape[0]
total_percentage = 1.0
total_mean = data['14070'].mean()
total_var = data['14070'].var()
results_df_rand = results_df_rand.append({'Strata' : 'Population',
                                'Count' : total_count,
                                'Percentage' : total_percentage,
                                'Mean' : total_mean,
                                'Variance' : variance}, ignore_index=True)

results_df_rand.to_excel('Table_for_rand_strats.xlsx')
results_df_rand
for strata in strata_labeles:
    rand_strata_data = combined_data[combined_data['strata'] == strata]
    count = rand_strata_data.shape[0]
    percentage = count / combined_data.shape[0]
    mean = rand_strata_data['14070'].mean()
    variance = rand_strata_data['14070'].var()
    results_df_rand = results_df_rand.append({'Strata': strata, 'Count': count, 'Percentage': percentage,
                                    'Mean': mean, 'Variance': variance}, ignore_index=True)

results_df_rand['real_mean'] = results_df_rand['Percentage']*results_df_rand['Mean']
results_df_rand
sum_real_mean_rand = results_df_rand['real_mean'].sum()
sum_real_mean_rand

results_df_rand['selcetive_variation'] = results_df_rand['Percentage']*results_df_rand['Variance']
sum_select_var_rand = results_df_rand['selcetive_variation'].sum()
sum_select_var_rand

selective_variance_rand = selective_variation(sum_select_var_rand, sum_rows_sample, total_count)
selective_variance_rand

deff = DEff(var_popul, sum_rows_sample, total_count, selective_variance_rand)
deff

# We also can calculate difference between real mean and mean 10% cohorts

y_mean_rand = np.array(results_df_rand['Mean'])
y_mean_rand
strats_rand = results_df_rand['Strata'].tolist()
strats_rand
difference_rand = [mean - sum_real_mean_rand for mean in y_mean_rand]
difference_rand

color_palette = plt.cm.viridis(np.linspace(0, 1, len(strats_rand)))

fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
x = np.arange(len(strats_rand))
for i in range(len(strats_rand)):
    ax.bar(x[i], difference_rand[i], color=color_palette[i], label=strats_rand[i])
ax.set_xlabel('Strat_rand')
ax.set_ylabel('Difference_rand')
ax.set_xticks(x)
ax.set_xticklabels(strats_rand, rotation=45, ha='right')
ax.set_title('Difference between Mean and True Mean for Randome sample')
ax.legend(loc="upper left")
plt.show()

# Let's build figure to show how many type of cohorts exist in every region

result = combined_data.groupby(['Субъект РФ', 'strata'], as_index=False).size()
result = result.rename(columns={'size': 'Number of strats'})
result
result = result[(result != 0).all(axis=1)]
result

result = result.groupby(['Субъект РФ', 'strata']).sum().unstack(fill_value=0)
result
