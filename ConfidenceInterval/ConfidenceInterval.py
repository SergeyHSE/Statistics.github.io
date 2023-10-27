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
        
        # The exact confidence interval (based on gamma distribution)
        a = sample_size
        scale = 1 / (sample_size * theta_hat)
        ci_3 = (stats.gamma.ppf(alpha/2, a, scale=scale), stats.gamma.ppf(1 - alpha/2, a, scale=scale))
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

theta = 1  
num_samples = 100
alpha = 0.05

sample_sizes = np.arange(0, 101, 10)

coverages_asymptotic = []
coverages_exact = []
mean_lengths_asymptotic = []
mean_lengths_exact = []

# we are gonna provide experiments for different sizes of samples
for sample_size in sample_sizes:
    results = confidence_experiment(theta, sample_size, num_samples, alpha)
    
    # Calculate share of interval coverage and mean lenth of intervals
    coverage_asymptotic = sum([1 for res in results if res[0][0] <= theta <= res[0][1]]) / num_samples
    coverage_exact = sum([1 for res in results if res[2][0] <= theta <= res[2][1]]) / num_samples

    mean_length_asymptotic = np.mean([res[0][1] - res[0][0] for res in results])
    mean_length_exact = np.mean([res[2][1] - res[2][0] for res in results])
 
    coverages_asymptotic.append(coverage_asymptotic)
    coverages_exact.append(coverage_exact)
    mean_lengths_asymptotic.append(mean_length_asymptotic)
    mean_lengths_exact.append(mean_length_exact)
    
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(sample_sizes, coverages_asymptotic, marker='o', label='Asнmptotic')
plt.plot(sample_sizes, coverages_exact, marker='o', label='Exat')
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

theta = 2 
num_samples = 100
alpha = 0.05

sample_sizes = np.arange(0, 101, 10)

coverages_asymptotic = []
coverages_exact = []
mean_lengths_asymptotic = []
mean_lengths_exact = []

for sample_size in sample_sizes:
    results = confidence_experiment(theta, sample_size, num_samples, alpha)
    
    coverage_asymptotic = sum([1 for res in results if res[0][0] <= theta <= res[0][1]]) / num_samples
    coverage_exact = sum([1 for res in results if res[2][0] <= theta <= res[2][1]]) / num_samples
    
    mean_length_asymptotic = np.mean([res[0][1] - res[0][0] for res in results])
    mean_length_exact = np.mean([res[2][1] - res[2][0] for res in results])
 
    coverages_asymptotic.append(coverage_asymptotic)
    coverages_exact.append(coverage_exact)
    mean_lengths_asymptotic.append(mean_length_asymptotic)
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
