# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 11:11:08 2018

@author: kazantseva
"""

import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import pandas as pd


lambda_param = 1 / 20
# exponential distribution with parameter lambda = 20
exp_dist = sts.expon(scale = 1 / lambda_param, loc = lambda_param)

sample = exp_dist.rvs(1000)

x = np.linspace(0, 100, 1000)

#probability density function
pdf = exp_dist.pdf(x)
#plt.plot(x, pdf, label='theoretical PDF')
#plt.hist(sample, label='empiric', normed = True, bins = 40)

#plot settings
#plt.ylabel('$f(x)$')
#plt.xlabel('$x$')
#plt.legend(loc='upper right')

#1000 samples with size = 5
n = 5
list_avg = [ exp_dist.rvs(n) for i in np.arange(1000)]
#emperic average of every sample 
list_avg = [ sum(arr) / len(arr) for i, arr in enumerate(list_avg) ]

x = np.linspace(15, 25, 1000)

# theoretical expected  = lambda ** (-1), theoretical dispersion  = lambda ** (-2)
norm_dist = sts.norm(1 / lambda_param, np.sqrt(1 / lambda_param ** 2 / 1000))
norm_pdf = norm_dist.pdf(x)
plt.hist(list_avg, normed = True, label = 'empiric')
plt.plot(x, norm_pdf, label='theoretical PDF')

plt.ylabel('$f(x)$')
plt.xlabel('$x$')
plt.legend(loc='upper right')