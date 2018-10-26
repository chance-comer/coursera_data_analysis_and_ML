# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 16:07:15 2018

@author: kazantseva
"""

import pandas as pd
import numpy as np
import scipy.stats as st 
from statsmodels.stats.multitest import multipletests 

data = pd.read_csv('gene_high_throughput_sequencing.csv')

normal = data[data['Diagnosis'] == 'normal']
early = data[data['Diagnosis'] == 'early neoplasia']
cancer = data[data['Diagnosis'] == 'cancer']

normal = normal.drop(['Patient_id', 'Diagnosis'], axis = 1)
early = early.drop(['Patient_id', 'Diagnosis'], axis = 1)
cancer = cancer.drop(['Patient_id', 'Diagnosis'], axis = 1)

early_normal = {}
early_cancer = {}

for i, col in enumerate(normal.columns):
   early_normal[col] = st.ttest_ind(normal[col], early[col], equal_var = False)
   early_cancer[col] = st.ttest_ind(early[col], cancer[col], equal_var = False)

pval_early_normal = pd.DataFrame([[k, early_normal[k].pvalue] for k in early_normal.keys()])
pval_early_cancer = pd.DataFrame([[k, early_cancer[k].pvalue] for k in early_cancer.keys()])

reject_early_normal, p_corrected_early_normal, a1_early_normal, a2_early_normal = multipletests(
    pval_early_normal[1], alpha = 0.05 / 2, method = 'holm')
reject_early_normal2, p_corrected_early_normal2, a1_early_normal2, a2_early_normal2 = multipletests(
    pval_early_normal[1], alpha = 0.05 / 2, method = 'fdr_bh')

reject_early_cancer, p_corrected_early_cancer, a1_early_cancer, a2_early_cancer = multipletests(
    pval_early_cancer[1], alpha = 0.05 / 2, method = 'holm')
reject_early_cancer2, p_corrected_early_cancer2, a1_early_cancer2, a2_early_cancer2 = multipletests(
    pval_early_cancer[1], alpha = 0.05 / 2, method = 'fdr_bh')

def foldchange(T, C):
  a1 = np.array(T)
  a2  = np.array(C)
  return [k[0] / k[1] if k[0] >= k[1] else - k[1] / k[0] for k in zip(a1, a2)]

p_corrected_early_cancer = pd.DataFrame(np.vstack((pval_early_cancer[0], p_corrected_early_cancer)).T)
p_corrected_early_normal = pd.DataFrame(np.vstack((pval_early_normal[0], p_corrected_early_normal)).T)
p_corrected_early_cancer2 = pd.DataFrame(np.vstack((pval_early_cancer[0], p_corrected_early_cancer2)).T)
p_corrected_early_normal2 = pd.DataFrame(np.vstack((pval_early_normal[0], p_corrected_early_normal2)).T)

significant_early_normal_holm = p_corrected_early_normal[p_corrected_early_normal[1] <  0.05 / 2]
significant_early_cancer_holm = p_corrected_early_cancer[p_corrected_early_cancer[1] <  0.05 / 2]

significant_early_normal_bh = p_corrected_early_normal2[p_corrected_early_normal2[1] <  0.05 / 2]
significant_early_cancer_bh = p_corrected_early_cancer2[p_corrected_early_cancer2[1] <  0.05 / 2]

fold_change_en_h = foldchange((early[significant_early_normal_holm[0]]).mean(), (normal[significant_early_normal_holm[0]]).mean())
fold_change_en_bh = foldchange((early[significant_early_normal_bh[0]]).mean(), (normal[significant_early_normal_bh[0]]).mean())

fold_change_ec_h = foldchange((cancer[significant_early_cancer_holm[0]]).mean(), (early[significant_early_cancer_holm[0]]).mean())
fold_change_ec_bh = foldchange((cancer[significant_early_cancer_bh[0]]).mean(), (early[significant_early_cancer_bh[0]]).mean())




