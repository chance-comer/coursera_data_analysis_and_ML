# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 10:43:09 2018

@author: kazantseva
"""

import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import zconfint
import scipy.stats as st
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests 

data = pd.read_csv('ab_browser_test.csv')

control = data[data['slot'] == 'control']
exp = data[data['slot'] == 'exp']

'''
1
'''
control_clicks = sum(control['n_clicks'])
exp_clicks = sum(exp['n_clicks'])

exp_per_control = exp_clicks * 100 / control_clicks
answer_1 = exp_per_control - 100

'''
2
'''
alpha = 0.05

control_count = len(control)
exp_count = len(exp)

control_bootstrap_indexes = np.random.randint(0, control_count, (500, control_count))
exp_bootstrap_indexes = np.random.randint(0, exp_count, (500, exp_count))

control_n_clicks = np.array(control['n_clicks'])
exp_n_cliks = np.array(exp['n_clicks'])

control_bootstrap = control_n_clicks[control_bootstrap_indexes]
exp_bootstrap = exp_n_cliks[exp_bootstrap_indexes]

control_means = np.array([x.mean() for x in control_bootstrap])
exp_means = np.array([x.mean() for x in exp_bootstrap])

control_medians = np.array([np.median(x) for x in control_bootstrap])
exp_medians = np.array([np.median(x) for x in exp_bootstrap])

boundaries_diff_means = np.percentile(control_means - exp_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
boundaries_diff_medians = np.percentile(control_medians - exp_medians, [100 * alpha / 2, 100 * (1 - alpha / 2)])

'''
4
'''
np.random.seed(0)
n_boot_samples = 500

control_bootstrap_indexes = np.random.randint(0, control_count, (n_boot_samples, int(np.floor(control_count / 10))))

control_n_clicks = np.array(control['n_clicks'])

control_bootstrap = control_n_clicks[control_bootstrap_indexes]

control_means = np.array([x.mean() for x in control_bootstrap])
control_dev = np.array([sum([(i - x[1])**2 for i in x[0]]) for x in zip(control_bootstrap, control_means)])

plt.subplot(1, 2, 1)
norm = st.probplot(control_means, dist="norm", plot=plt)
plt.subplot(1, 2, 2)
chi = st.probplot(control_dev, dist="chi2", sparams=(n_boot_samples-1), plot=plt)

'''
5
'''
control_slice = control[['userID', 'n_clicks']]
control_user = control_slice.groupby('userID').sum()

exp_slice = exp[['userID', 'n_clicks']]
exp_user = exp_slice.groupby('userID').sum()

manna = st.mannwhitneyu(control_user, exp_user)

'''
6
'''
unique_browser = control.groupby('browser').count().index

exp_control = {}

for i, col in enumerate(unique_browser):
   exp_control[col] = st.mannwhitneyu(control[control['browser'] == col]['n_clicks'], exp[exp['browser'] == col]['n_clicks'])

pval_exp_control = pd.DataFrame([[k, exp_control[k].pvalue] for k in exp_control.keys()])

reject_exp_control, p_corrected_exp_control, a1_exp_control, a2_exp_control = multipletests(
    pval_exp_control[1], alpha = 0.05, method = 'holm')
  
#control_browser_slice = control['']
'''
7
'''
control_nonclicks = {}
exp_nonclicks = {}

control_brq_slice = control[['browser', 'n_queries']]
control_brnq_slice = control[['browser', 'n_nonclk_queries']]
control_brq = control_brq_slice.groupby('browser').sum()
control_brnq = control_brnq_slice.groupby('browser').sum()

exp_brq_slice = exp[['browser', 'n_queries']]
exp_brnq_slice = exp[['browser', 'n_nonclk_queries']]
exp_brq = exp_brq_slice.groupby('browser').sum()
exp_brnq = exp_brnq_slice.groupby('browser').sum()
  #control_nonclicks[col] = 
a71 = [g[1] / g[0] * 100 for g in zip(control_brq['n_queries'], control_brnq['n_nonclk_queries'])]
a72 = [g[1] / g[0] * 100 for g in zip(exp_brq['n_queries'], exp_brnq['n_nonclk_queries'])]