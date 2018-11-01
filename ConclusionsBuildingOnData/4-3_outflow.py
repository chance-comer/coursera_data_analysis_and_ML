# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 10:19:25 2018

@author: kazantseva
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, pearsonr, spearmanr, probplot
import matplotlib.pyplot as plt

data = pd.read_csv('churn_analysis.csv')

control = data[data['treatment'] == 1]
control['churn'] = [0 if c == 'False.' else 1 for c in control['churn']]
states = control['state'].value_counts()

chi2_state_churn_res = {}
chi2_state_churn_res_with_corr = {}
count_0_1 = {}
fisher_state_churn_res = {}
'''
for i, st in enumerate(states.index):
  for j, st2 in enumerate(states.index):
    if j <= i:
      continue
    subt_stF = len(control[control['state'] == st][control['churn'] == 0])
    subt_stT = len(control[control['state'] == st][control['churn'] == 1])
    subt_st2F = len(control[control['state'] == st2][control['churn'] == 0])
    subt_st2T = len(control[control['state'] == st2][control['churn'] == 1])
    chi2 = chi2_contingency([[subt_stF, subt_stT],[subt_st2F, subt_st2T]], correction = False)
    chi2_state_churn_res[st + '+' + st2] = chi2[1]
    chi2_corr = chi2_contingency([[subt_stF, subt_stT],[subt_st2F, subt_st2T]], correction = True)
    chi2_state_churn_res_with_corr[st + '+' + st2] = chi2_corr[1]
    fisher_state_churn_res[st + '+' + st2] = fisher_exact([[subt_stF, subt_stT],[subt_st2F, subt_st2T]])[1]
  count_0_1[st] = [subt_stF, subt_stT]

answer_1 = len(list(filter(lambda m: chi2_state_churn_res[m] < 0.05, chi2_state_churn_res)))
answer_3 = len(list(filter(lambda m: chi2_state_churn_res_with_corr[m] < 0.05, chi2_state_churn_res_with_corr)))
answer_4 = len(list(filter(lambda m: fisher_state_churn_res[m] < 0.05, fisher_state_churn_res)))
'''
day_calls = data['day_calls']
mes_estim = data['mes_estim']
p_c = pearsonr(day_calls, mes_estim)
s_c = spearmanr(day_calls, mes_estim)

confusions = []
for i, st in enumerate(states.index):
    subt_stF = len(control[control['state'] == st][control['churn'] == 0])
    subt_stT = len(control[control['state'] == st][control['churn'] == 1])
    confusions.append(subt_stF)
    confusions.append(subt_stT)    
confusion_matrix = np.array(confusions).reshape(len(states), 2)

def cramers_stat(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    return np.sqrt(chi2 / (n*(min(confusion_matrix.shape)-1)))

result = cramers_stat(confusion_matrix)

treat_0 = data[data['treatment'] == 0]
treat_2 = data[data['treatment'] == 2]
treat_0['churn'] = [0 if c == 'False.' else 1 for c in treat_0['churn']]
treat_2['churn'] = [0 if c == 'False.' else 1 for c in treat_2['churn']]

fisher_01 = fisher_exact([[sum(control['churn']), len(control['churn']) - sum(control['churn'])],\
                           [sum(treat_0['churn']), len(treat_0['churn']) - sum(treat_0['churn'])]])
fisher21 = fisher_exact([[sum(control['churn']), len(control['churn']) - sum(control['churn'])],\
                           [sum(treat_2['churn']), len(treat_2['churn']) - sum(treat_2['churn'])]])