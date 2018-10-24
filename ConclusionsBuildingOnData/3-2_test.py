# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 10:45:54 2018

@author: kazantseva
"""

import pandas as pd
import scipy.stats as st
import numpy as np

data = pd.read_csv('water.txt', sep = '\t')

answer_1 = data.corr()
answer_2 = data.corr(method = 'spearman')

south = data[data['location'] == 'South']
north = data[data['location'] == 'North']

answer_3 = min(abs(south.corr()['mortality']['hardness']), abs(north.corr()['mortality']['hardness']))

a = 718
b = 203
c = 515
d = 239

answer_4 = (a * d - b * c) / ((a + b) * (a + c) * (c + d) * (b + d)) ** 0.5

answer_5 = st.chi2_contingency([[a, b], [c, d]])

alpha = 0.05

woman = b / (b + a)
man = d / (d + c)

z = st.norm.ppf(1 - alpha / 2)

left_b = man - woman - z * np.sqrt(woman * (1 - woman) / (b + a) + man * (1 - man) / (d + c))

#answer_7 = st.chisquare([woman, man], [0.5, 0.5], ddof = 0)
fem = np.append(np.ones(203), np.zeros(718))
mal = np.append(np.ones(239), np.zeros(515))

P = (woman * (b + a) + man * (d + c)) / (a + c + b + d)

z_7 = (man - woman) / (P * (1 - P) * (1 / (b + a) + 1 / (d + c))) ** 0.5

lvl_7 = 2 * (1 - st.norm.cdf(abs(z_7)))

answer_8 = st.chi2_contingency([[197, 111, 33], [382, 685, 331], [110, 342, 333]])

answer_10 = np.sqrt(answer_8[0] / (2 * (197 + 111 + 33 + 382 + 685 + 331 + 110 + 342 + 333)))