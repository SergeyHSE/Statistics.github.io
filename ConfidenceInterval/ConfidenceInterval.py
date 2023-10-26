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
