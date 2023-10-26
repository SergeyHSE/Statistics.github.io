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
