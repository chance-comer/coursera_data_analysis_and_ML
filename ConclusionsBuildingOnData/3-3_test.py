# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 17:10:37 2018

@author: kazantseva
"""

import pandas as pd
from scipy import stats
from statsmodels.stats.descriptivestats import sign_test
from statsmodels.sandbox.stats.multicomp import multipletests 

data = pd.read_csv('AUCs.txt', sep = '\t')
data = data.drop('Unnamed: 0', axis = 1)

#answer 2, 3, 4
answer_2 = {}

for i, col in enumerate(data.columns):
  for j, col2 in enumerate(data.columns):
    if j <= i:
      continue
    answer_2[col + '+' + col2] = stats.wilcoxon(data[col], data[col2])

pval = pd.DataFrame([[k, answer_2[k].pvalue] for k in answer_2.keys()])

#answer 5
reject, p_corrected, a1, a2 = multipletests(pval[1], alpha = 0.05, method = 'holm')

#answer 6
reject2, p_corrected2, a12, a22 = multipletests(pval[1], alpha = 0.05, method = 'fdr_bh') 

