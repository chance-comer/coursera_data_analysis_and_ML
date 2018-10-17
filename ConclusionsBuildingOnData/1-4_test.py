# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 16:08:25 2018

@author: kazantseva
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, binned_statistic_2d, chisquare
from statsmodels.stats.proportion import proportion_confint, samplesize_confint_proportion

# H0 - уровень стресса не отличается. Средние равны.
p_75 = proportion_confint(75, 100, method = 'normal')
p_60 = proportion_confint(60, 100, method = 'normal')

count1 = 100
count2 = 100

s_count1 = 60
s_count2 = 75

p1 = s_count1 / count1
p2 = s_count2 / count2

p_difference = p2 - p1

alpha = 0.05

lvl = 2 * norm.sf(p_difference)

z = norm.ppf(1 - alpha / 2)

p_difference_lb = p_difference - z * (p1 * (1 - p1) / s_count1 + p2 * (1 - p2) / s_count2) ** 0.5
p_difference_rb = p_difference + z * (p1 * (1 - p1) / s_count1 + p2 * (1 - p2) / s_count2) ** 0.5

data = pd.read_csv('pines.txt', sep = '\t')

hist = binned_statistic_2d(data.iloc[:, 0], data.iloc[:, 1], None, 'count', [5, 5])
lf = hist.statistic.flatten()
answer = chisquare(lf, [hist.statistic.flatten().sum() / 25] * len(lf))
