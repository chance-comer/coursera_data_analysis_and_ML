# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 10:44:44 2018

@author: kazantseva
"""

from scipy.stats import norm
import numpy as np

asp_count = 11037
placebo_count = 11034

infarct_asp_count = 104
infarct_placebo_count = 189

p_asp = infarct_asp_count / asp_count
p_placebo = infarct_placebo_count / placebo_count

p_difference = p_placebo - p_asp

alpha = 0.05

z = norm.ppf(1 - alpha / 2)

p_difference_lb = p_difference - z * (p_asp * (1 - p_asp) / asp_count + p_placebo * (1 - p_placebo) / placebo_count) ** 0.5
p_difference_rb = p_difference + z * (p_asp * (1 - p_asp) / asp_count + p_placebo * (1 - p_placebo) / placebo_count) ** 0.5

asp_odds = p_asp / (1 - p_asp)
placebo_odds = p_placebo / (1 - p_placebo)

odds_ratio = placebo_odds / asp_odds

asp_sample = np.array([1 if i < infarct_asp_count else 0 for i in range(asp_count)])
placebo_sample = np.array([1 if i < infarct_placebo_count else 0 for i in range(placebo_count)])

np.random.seed(0)

asp_bootstrap_indexes = np.random.randint(0, asp_count, (1000, asp_count))
placebo_bootstrap_indexes = np.random.randint(0, placebo_count, (1000, placebo_count))

asp_bootstrap = asp_sample[asp_bootstrap_indexes]
placebo_bootstrap = placebo_sample[placebo_bootstrap_indexes]

asp_bootstrap_odds =  list(map(lambda x: (x.sum() / len(x)) / (1 - x.sum() / len(x)) , asp_bootstrap))
placebo_bootstrap_odds =  list(map(lambda x: (x.sum() / len(x)) / (1 - x.sum() / len(x)) , placebo_bootstrap))

odds_bootstrap_ratio = list(map(lambda x: x[1] / x[0], list(zip(asp_bootstrap_odds, placebo_bootstrap_odds))))

answer = np.percentile(odds_bootstrap_ratio, [100 * alpha / 2, 100 * (1 - alpha / 2)])

