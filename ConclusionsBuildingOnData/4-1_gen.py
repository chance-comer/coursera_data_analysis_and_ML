# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 16:07:15 2018

@author: kazantseva
"""

import pandas as pd
import numpy as np
import scipy.stats as st 
from statsmodels.sandbox.stats.multicomp import multipletests 

data = pd.read_csv('gene_high_throughput_sequencing.csv')

normal = data[data['Diagnosis'] == 'normal']
early = data[data['Diagnosis'] == 'early neoplasia']
cancer = data[data['Diagnosis'] == 'cancer']

normal = normal.drop(['Patient_id', 'Diagnosis'], axis = 1)
early = early.drop(['Patient_id', 'Diagnosis'], axis = 1)
cancer = cancer.drop(['Patient_id', 'Diagnosis'], axis = 1)

easy_normal = {}
normal_cancer = {}

for i, col in enumerate(normal.columns):
   easy_normal[col] = st.ttest_ind(normal[col], early[col], equal_var = False)
   normal_cancer[col] = st.ttest_ind(normal[col], cancer[col], equal_var = False)

pval_easy_normal = pd.DataFrame([[k, easy_normal[k].pvalue] for k in easy_normal.keys()])
pval_normal_cancer = pd.DataFrame([[k, normal_cancer[k].pvalue] for k in normal_cancer.keys()])

reject_easy_normal, p_corrected_easy_normal, a1_easy_normal, a2_easy_normal = multipletests(
    pval_easy_normal[1], alpha = 0.05 / 2, method = 'holm')
reject_easy_normal2, p_corrected_easy_normal2, a1_easy_normal2, a2_easy_normal2 = multipletests(
    pval_easy_normal[1], alpha = 0.05 / 2, method = 'fdr_bh')

reject_normal_cancer, p_corrected_normal_cancer, a1_normal_cancer, a2_normal_cancer = multipletests(
    pval_normal_cancer[1], alpha = 0.05 / 2, method = 'holm')
reject_normal_cancer2, p_corrected_normal_cancer2, a1_normal_cancer2, a2_normal_cancer2 = multipletests(
    pval_normal_cancer[1], alpha = 0.05 / 2, method = 'fdr_bh')

def foldchange(T, C):
  if T >= C:
    return T / C
  else:
    return - C / T

p_corrected_normal_cancer = pd.DataFrame(np.vstack((pval_easy_normal[0], p_corrected_normal_cancer)).T)
p_corrected_easy_normal = pd.DataFrame(np.vstack((pval_normal_cancer[0], p_corrected_easy_normal)).T)
p_corrected_normal_cancer2 = pd.DataFrame(np.vstack((pval_easy_normal[0], p_corrected_normal_cancer2)).T)
p_corrected_easy_normal2 = pd.DataFrame(np.vstack((pval_normal_cancer[0], p_corrected_easy_normal2)).T)

significant_easy_normal_holm = p_corrected_easy_normal[p_corrected_easy_normal[1] < 0.05]
significant_normal_cancer_holm = p_corrected_normal_cancer[p_corrected_normal_cancer[1] < 0.05]

significant_easy_normal_bh = p_corrected_easy_normal2[p_corrected_easy_normal2[1] < 0.05]
significant_normal_cancer_bh = p_corrected_normal_cancer2[p_corrected_normal_cancer2[1] < 0.05]

fold_change_en_h = foldchange((early[significant_easy_normal_holm[0]]).mean(), normal[significant_easy_normal_holm[0]]).mean())


