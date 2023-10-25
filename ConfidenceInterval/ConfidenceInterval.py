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
