"""
Let' model a lot of experiments with the same conditions.
We are gonna generate about 20 samples with size 10 and calculate the borders all of them
and check H-hypothesis
"""
import numpy as np
import scipy.stats as stats
import pandas as pd

def confidence_experiment(theta, sample_size=10, num_samples=20, alpha=0.05):
    results = []
    
    for _ in range(num_samples):
        # Generate samples
        data = np.random.exponential(scale=1/theta, size=sample_size)
        theta_hat = sample_size / np.sum(data)

        # Asymptotic confidence interval 1
        se_1 = theta_hat / np.sqrt(sample_size)
        ci_1 = (theta_hat - stats.norm.ppf(1 - alpha/2) * se_1, theta_hat + stats.norm.ppf(1 - alpha/2) * se_1)
        
        # (для theta^2)
        se_2 = (2 * theta_hat**2) / np.sqrt(sample_size)
        ci_2 = (theta_hat**2 - stats.norm.ppf(1 - alpha/2) * se_2, theta_hat**2 + stats.norm.ppf(1 - alpha/2) * se_2)
        
        # Exact confidence interval (using gamma distribution)
        lower_gamma = stats.gamma.ppf(alpha/2, sample_size, scale=1/theta_hat)
        upper_gamma = stats.gamma.ppf(1 - alpha/2, sample_size, scale=1/theta_hat)
        ci_3 = (sample_size / upper_gamma, sample_size / lower_gamma)
        
        # Results
        crit_1 = stats.norm.interval(1 - alpha, loc=0, scale=1)[0] <= (theta_hat - theta) / se_1 <= stats.norm.interval(1 - alpha, loc=0, scale=1)[1]
        crit_2 = stats.norm.interval(1 - alpha, loc=0, scale=1)[0] <= (theta_hat**2 - theta**2) / se_2 <= stats.norm.interval(1 - alpha, loc=0, scale=1)[1]
        crit_3 = ci_3[0] <= theta <= ci_3[1]
        
        results.append([ci_1, ci_2, ci_3, crit_1, crit_2, crit_3])
    
    return results

theta = 1
sample_size = 10
num_samples = 20
alpha = 0.05

results = confidence_experiment(theta, sample_size, num_samples, alpha)
df = pd.DataFrame(results, columns=["Interval_1", "Interval_2", "Interval_3", 
                                    "One", "Asymptotic", "Exact"])

df = df.apply(lambda x: [f"({val[0]:.2f}, {val[1]:.2f})" if isinstance(val, tuple) else val for val in x])

print(df)

# Repeat the same actions for wrong H:0

theta = 2 

results = confidence_experiment(theta, sample_size, num_samples, alpha)
df = pd.DataFrame(results, columns=["Interval_1", "Interval_2", "Interval_3", 
                                    "One", "Asymptotic", "Exact"])

df = df.apply(lambda x: [f"({val[0]:.2f}, {val[1]:.2f})" if isinstance(val, tuple) else val for val in x])

print(df)

"""
In the next step we are gonna show the relationship berween confidance interval and size of samples.
Also we need estimate share of interval coverage.
"""

import matplotlib.pyplot as plt

theta = 1  
num_samples = 100
alpha = 0.05
sample_sizes = np.arange(0, 101, 10)

coverages_asymptotic_1 = []  # To store coverage proportions for the first asymptotic interval
coverages_asymptotic_2 = []  # To store coverage proportions for the second asymptotic interval
coverages_exact = []  # To store coverage proportions for the exact interval
mean_lengths_asymptotic_1 = []  # To store mean lengths for the first asymptotic interval
mean_lengths_asymptotic_2 = []  # To store mean lengths for the second asymptotic interval
mean_lengths_exact = []  # To store mean lengths for the exact interval

# we are gonna provide experiments for different sizes of samples
for sample_size in sample_sizes:
    results = confidence_experiment(theta, sample_size, num_samples, alpha)
    
    # Calculate share of interval coverage and mean lenth of intervals
    coverage_asymptotic_1 = sum([1 for res in results if res[0][0] <= theta <= res[0][1]]) / num_samples
    coverage_asymptotic_2 = sum([1 for res in results if res[1][0] <= theta**2 <= res[1][1]]) / num_samples
    coverage_exact = sum([1 for res in results if res[2][0] <= theta <= res[2][1]]) / num_samples

    mean_length_asymptotic_1 = np.mean([res[0][1] - res[0][0] for res in results])
    mean_length_asymptotic_2 = np.mean([res[1][1] - res[1][0] for res in results])
    mean_length_exact = np.mean([res[2][1] - res[2][0] for res in results])

    coverages_asymptotic_1.append(coverage_asymptotic_1)
    coverages_asymptotic_2.append(coverage_asymptotic_2)
    coverages_exact.append(coverage_exact)
    mean_lengths_asymptotic_1.append(mean_length_asymptotic_1)
    mean_lengths_asymptotic_2.append(mean_length_asymptotic_2)
    mean_lengths_exact.append(mean_length_exact)
    
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(sample_sizes, coverages_asymptotic_1, marker='o', label='Asymptotic 1')
plt.plot(sample_sizes, coverages_asymptotic_2, marker='o', label='Asymptotic 2')
plt.plot(sample_sizes, coverages_exact, marker='o', label='Exact')
plt.xlabel('Sample Size')
plt.ylabel('Coverage Proportion')
plt.legend()
plt.grid()
plt.subplot(1, 2, 2)
plt.plot(sample_sizes, mean_lengths_asymptotic_1, marker='o', label='Asymptotic 1')
plt.plot(sample_sizes, mean_lengths_asymptotic_2, marker='o', label='Asymptotic 2')
plt.plot(sample_sizes, mean_lengths_exact, marker='o', label='Exact')
plt.xlabel('Sample Size')
plt.ylabel('Mean Length of Interval')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

theta = 2 
num_samples = 100
alpha = 0.05

sample_sizes = np.arange(0, 101, 10)

coverages_asymptotic_1 = []  # To store coverage proportions for the first asymptotic interval
coverages_asymptotic_2 = []  # To store coverage proportions for the second asymptotic interval
coverages_exact = []  # To store coverage proportions for the exact interval
mean_lengths_asymptotic_1 = []  # To store mean lengths for the first asymptotic interval
mean_lengths_asymptotic_2 = []  # To store mean lengths for the second asymptotic interval
mean_lengths_exact = []  # To store mean lengths for the exact interval

for sample_size in sample_sizes:
    results = confidence_experiment(theta, sample_size, num_samples, alpha)
    
    coverage_asymptotic_1 = sum([1 for res in results if res[0][0] <= theta <= res[0][1]]) / num_samples
    coverage_asymptotic_2 = sum([1 for res in results if res[1][0] <= theta**2 <= res[1][1]]) / num_samples
    coverage_exact = sum([1 for res in results if res[2][0] <= theta <= res[2][1]]) / num_samples

    mean_length_asymptotic_1 = np.mean([res[0][1] - res[0][0] for res in results])
    mean_length_asymptotic_2 = np.mean([res[1][1] - res[1][0] for res in results])
    mean_length_exact = np.mean([res[2][1] - res[2][0] for res in results])

    coverages_asymptotic_1.append(coverage_asymptotic_1)
    coverages_asymptotic_2.append(coverage_asymptotic_2)
    coverages_exact.append(coverage_exact)
    mean_lengths_asymptotic_1.append(mean_length_asymptotic_1)
    mean_lengths_asymptotic_2.append(mean_length_asymptotic_2)
    mean_lengths_exact.append(mean_length_exact)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(sample_sizes, coverages_asymptotic, marker='o', label='Asymtotic')
plt.plot(sample_sizes, coverages_exact, marker='o', label='Exact')
plt.xlabel('Size of sample')
plt.ylabel('Share of coverage')
plt.legend()
plt.grid()
plt.subplot(1, 2, 2)
plt.plot(sample_sizes, mean_lengths_asymptotic, marker='o', label='Asymptotic')
plt.plot(sample_sizes, mean_lengths_exact, marker='o', label='Exact')
plt.xlabel('Size of sample')
plt.ylabel('Mean lenth of interval')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

"""
Let's define probabolity of type 1 error and build confidance interval
"""

alpha = 0.05
num_simulations = 1000
sample_sizes = range(1, 101, 5) 
theta = 1

error_rates_criterion_1 = []
error_rates_criterion_2 = []
error_rates_criterion_3 = []

for sample_size in sample_sizes:
    errors_criterion_1 = 0  
    errors_criterion_2 = 0  
    errors_criterion_3 = 0 
    
    for _ in range(num_simulations):
        data = np.random.exponential(scale=1/theta, size=sample_size)

        theta_hat = sample_size / np.sum(data)

        se_1 = theta_hat / np.sqrt(sample_size)
        se_2 = (2 * theta_hat**2) / np.sqrt(sample_size)
        
        crit_1 = stats.norm.interval(1 - alpha, loc=0, scale=1)[0] <= (theta_hat - theta) / se_1 <= stats.norm.interval(1 - alpha, loc=0, scale=1)[1]
        crit_2 = stats.norm.interval(1 - alpha, loc=0, scale=1)[0] <= (theta_hat**2 - theta**2) / se_2 <= stats.norm.interval(1 - alpha, loc=0, scale=1)[1]

        a = sample_size
        scale = 1 / (sample_size * theta_hat)
        left_exact = stats.gamma.ppf(alpha/2, a, scale=scale)
        right_exact = stats.gamma.ppf(1 - alpha/2, a, scale=scale)
        crit_3 = left_exact <= theta <= right_exact
        
        if crit_1:
            errors_criterion_1 += 1
        if crit_2:
            errors_criterion_2 += 1
        if crit_3:
            errors_criterion_3 += 1

    error_rate_criterion_1 = errors_criterion_1 / num_simulations
    error_rate_criterion_2 = errors_criterion_2 / num_simulations
    error_rate_criterion_3 = errors_criterion_3 / num_simulations
    
    error_rates_criterion_1.append(error_rate_criterion_1)
    error_rates_criterion_2.append(error_rate_criterion_2)
    error_rates_criterion_3.append(error_rate_criterion_3)

mean_error_rate_criterion_1 = np.mean(error_rates_criterion_1)
mean_error_rate_criterion_2 = np.mean(error_rates_criterion_2)
mean_error_rate_criterion_3 = np.mean(error_rates_criterion_3)

std_error_rate_criterion_1 = np.std(error_rates_criterion_1, ddof=1)
std_error_rate_criterion_2 = np.std(error_rates_criterion_2, ddof=1)
std_error_rate_criterion_3 = np.std(error_rates_criterion_3, ddof=1)

z_critical = stats.norm.ppf(1 - alpha / 2)
margin_of_error_criterion_1 = z_critical * (std_error_rate_criterion_1 / np.sqrt(num_simulations))
margin_of_error_criterion_2 = z_critical * (std_error_rate_criterion_2 / np.sqrt(num_simulations))
margin_of_error_criterion_3 = z_critical * (std_error_rate_criterion_3 / np.sqrt(num_simulations))

confidence_interval_criterion_1 = (mean_error_rate_criterion_1 - margin_of_error_criterion_1,
                                   mean_error_rate_criterion_1 + margin_of_error_criterion_1)
confidence_interval_criterion_2 = (mean_error_rate_criterion_2 - margin_of_error_criterion_2,
                                   mean_error_rate_criterion_2 + margin_of_error_criterion_2)
confidence_interval_criterion_3 = (mean_error_rate_criterion_3 - margin_of_error_criterion_3,
                                   mean_error_rate_criterion_3 + margin_of_error_criterion_3)

print("Mean of type 1 error (criterion 1):", mean_error_rate_criterion_1)
print("Confidence interval (criterion 1):", confidence_interval_criterion_1)

print("Mean of type error (criterion 2):", mean_error_rate_criterion_2)
print("Confidence interval (criterion 2):", confidence_interval_criterion_2)

print("Mean of type error (criterion 3):", mean_error_rate_criterion_3)
print("Confidenece interval (criterion 3):", confidence_interval_criterion_3)

"""
We are gonna estimate statistic power depending on the value of theta and size of sample
"""

alpha = 0.05
num_simulations = 1000

theta_values = [1, 2, 3, 4, 5, 6]  
sample_sizes = [1, 20, 40, 60, 80, 100]

power_criterion_1 = []
power_criterion_2 = []
power_criterion_3 = []

for theta in theta_values:
    power_criterion_1_theta = []
    power_criterion_2_theta = []
    power_criterion_3_theta = []

    for sample_size in sample_sizes:
        rejections_criterion_1 = 0
        rejections_criterion_2 = 0
        rejections_criterion_3 = 0

        for _ in range(num_simulations):
            data = np.random.exponential(scale=1 / theta, size=sample_size)

            theta_hat = sample_size / np.sum(data)

            se_1 = theta_hat / np.sqrt(sample_size)
            se_2 = (2 * theta_hat ** 2) / np.sqrt(sample_size)

            crit_1 = stats.norm.interval(1 - alpha, loc=0, scale=1)[0] <= (theta_hat - theta) / se_1 <= stats.norm.interval(
                1 - alpha, loc=0, scale=1)[1]
            crit_2 = stats.norm.interval(1 - alpha, loc=0, scale=1)[0] <= (theta_hat ** 2 - theta ** 2) / se_2 <= stats.norm.interval(
                1 - alpha, loc=0, scale=1)[1]
            crit_3 = (1 / stats.gamma.ppf(alpha / 2, sample_size, scale=1 / (sample_size * theta_hat))) <= theta <= (
                    1 / stats.gamma.ppf(1 - alpha / 2, sample_size, scale=1 / (sample_size * theta_hat)))

            if crit_1:
                rejections_criterion_1 += 1
            if crit_2:
                rejections_criterion_2 += 1
            if crit_3:
                rejections_criterion_3 += 1

        power_criterion_1_theta.append(rejections_criterion_1 / num_simulations)
        power_criterion_2_theta.append(rejections_criterion_2 / num_simulations)
        power_criterion_3_theta.append(rejections_criterion_3 / num_simulations)

    power_criterion_1.append(power_criterion_1_theta)
    power_criterion_2.append(power_criterion_2_theta)
    power_criterion_3.append(power_criterion_3_theta)

df_criterion_1 = pd.DataFrame(power_criterion_1, columns=sample_sizes, index=theta_values)
df_criterion_2 = pd.DataFrame(power_criterion_2, columns=sample_sizes, index=theta_values)
df_criterion_3 = pd.DataFrame(power_criterion_3, columns=sample_sizes, index=theta_values)

print("The power of the first criterion:")
print(df_criterion_1)

print("\nThe power of the asymptotic criterion:")
print(df_criterion_2)

print("\nThe power of the exact criterion:")
print(df_criterion_3)

"""
The power of the first criterion:
     1      20     40     60     80     100
1  0.947  0.951  0.932  0.955  0.957  0.947
2  0.938  0.954  0.951  0.932  0.950  0.955
3  0.950  0.956  0.950  0.966  0.951  0.959
4  0.950  0.962  0.954  0.951  0.945  0.954
5  0.954  0.957  0.959  0.964  0.955  0.941
6  0.951  0.950  0.940  0.948  0.936  0.949

The power of the asymptotic criterion:
     1      20     40     60     80     100
1  0.884  0.943  0.936  0.947  0.960  0.946
2  0.880  0.940  0.942  0.927  0.952  0.954
3  0.903  0.944  0.937  0.958  0.952  0.959
4  0.890  0.954  0.953  0.934  0.944  0.949
5  0.894  0.935  0.958  0.962  0.949  0.941
6  0.890  0.944  0.952  0.950  0.938  0.947

The power of the exact criterion:
   1    20   40   60   80   100
1  0.0  0.0  0.0  0.0  0.0  0.0
2  0.0  0.0  0.0  0.0  0.0  0.0
3  0.0  0.0  0.0  0.0  0.0  0.0
4  0.0  0.0  0.0  0.0  0.0  0.0
5  0.0  0.0  0.0  0.0  0.0  0.0
6  0.0  0.0  0.0  0.0  0.0  0.0
"""

fig, axs = plt.subplots(3, figsize=(10, 15))
for i, n in enumerate(sample_sizes):
    axs[0].plot(theta_values, power_criterion_1[i], label=f"n = {n}", lw=3)
axs[0].set_xlabel("Parametr $\\theta$")
axs[0].set_ylabel("Power")
axs[0].set_title("The power of the first criterion")
axs[0].legend()
for i, n in enumerate(sample_sizes):
    axs[1].plot(theta_values, power_criterion_2[i], label=f"n = {n}", lw=3)
axs[1].set_xlabel("Parametr $\\theta$")
axs[1].set_ylabel("Power")
axs[1].set_title("The power of the asymptotic criterion")
axs[1].legend()
for i, n in enumerate(sample_sizes):
    axs[2].plot(theta_values, power_criterion_3[i], label=f"n = {n}", lw=3)
axs[2].set_xlabel("Parametr $\\theta$")
axs[2].set_ylabel("Power")
axs[2].set_title("The power of the exact criterion")
axs[2].legend()
plt.tight_layout()
plt.show()
