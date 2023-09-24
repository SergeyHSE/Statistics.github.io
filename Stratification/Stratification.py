# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 22:39:56 2023

@author: SergeyHSE

We are gonna extract colums kike region, revenue and pig farming cost,
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
