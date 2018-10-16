# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 17:42:54 2018

@author: kazantseva
"""

from statsmodels.stats.proportion import proportion_confint, samplesize_confint_proportion
import numpy as np 
import matplotlib.pyplot as plt

prop = proportion_confint(1, 50, method = 'wilson')

count = samplesize_confint_proportion(0.02, 0.01)

ps = np.linspace(0, 1, 500)

cs = [samplesize_confint_proportion(p, 0.01) for p in ps]

plt.plot(ps, cs)