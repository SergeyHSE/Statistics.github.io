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
