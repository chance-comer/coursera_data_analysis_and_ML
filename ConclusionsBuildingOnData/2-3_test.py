# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 16:43:00 2018

@author: kazantseva
"""
import numpy as np
import pandas as pd
import itertools

from scipy import stats
from statsmodels.stats.descriptivestats import sign_test
from statsmodels.stats.weightstats import zconfint
import matplotlib.pyplot as plt

'''
answer 4
'''

m0 = 200
a = np.array([49,58,75,110,112,132,151,276,281,362])
answer4 = stats.wilcoxon(a - m0)

'''
answer 5
'''
n1 = np.array([22,22,15,13,19,19,18,20,21,13,13,15])
n2 = np.array([17,18,18,15,12,4,14,15,10])

answer5 = stats.mannwhitneyu(n1, n2)

'''
answer 6
'''
alpha = 0.05

data = pd.read_csv('challenger.txt', sep = '\t')
temper_incident = np.array(data[data['Incident'] == 1]['Temperature'])
#temper_incident.index = range(0, len(temper_incident))
temper_notincident = np.array(data[data['Incident'] == 0]['Temperature'])
#temper_notincident.index = range(0, len(temper_notincident))

inc_count = len(temper_incident)
notinc_count = len(temper_notincident)

np.random.seed(0)

inc_bootstrap_indexes = np.random.randint(0, inc_count, (1000, inc_count))
notinc_bootstrap_indexes = np.random.randint(0, notinc_count, (1000, notinc_count))

inc_bootstrap = temper_incident[inc_bootstrap_indexes]
notinc_bootstrap = temper_notincident[notinc_bootstrap_indexes]

inc_means = np.array([x.mean() for x in inc_bootstrap])
notinc_means = np.array([x.mean() for x in notinc_bootstrap])

boundaries = np.percentile(notinc_means - inc_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])

'''
answer 7
'''
def permutation_t_stat_ind(sample1, sample2):
    return np.mean(sample1) - np.mean(sample2)
  
def get_random_combinations(n1, n2, max_combinations):
  index = list(range(n1 + n2))
  indices = set([tuple(index)])
  for i in range(max_combinations - 1):
      np.random.shuffle(index)
      indices.add(tuple(index))
  return [(index[:n1], index[n1:]) for index in indices]

def permutation_zero_dist_ind(sample1, sample2, max_combinations = None):
    joined_sample = np.hstack((sample1, sample2))
    n1 = len(sample1)
    n = len(joined_sample)
    
    if max_combinations:
        indices = get_random_combinations(n1, len(sample2), max_combinations)
    else:
        indices = [(list(index), filter(lambda i: i not in index, range(n))) \
                    for index in itertools.combinations(range(n), n1)]
    
    distr = [joined_sample[list(i[0])].mean() - joined_sample[list(i[1])].mean() \
             for i in indices]
    return distr

def permutation_test(sample, mean, max_permutations = None, alternative = 'two-sided'):
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")
    
    t_stat = permutation_t_stat_ind(sample, mean)
    
    zero_distr = permutation_zero_dist_ind(sample, mean, max_permutations)
    
    if alternative == 'two-sided':
        return sum([1. if abs(x) >= abs(t_stat) else 0. for x in zero_distr]) / len(zero_distr)
    
    if alternative == 'less':
        return sum([1. if x <= t_stat else 0. for x in zero_distr]) / len(zero_distr)

    if alternative == 'greater':
        return sum([1. if x >= t_stat else 0. for x in zero_distr]) / len(zero_distr)

np.random.seed(0)

answer7 = permutation_test(temper_incident, temper_notincident, max_permutations = 10000)
