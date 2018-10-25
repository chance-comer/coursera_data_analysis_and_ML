# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 11:15:27 2018

@author: kazantseva
"""

import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms

data = pd.read_csv('botswana.tsv', sep = '\t')

'''
answer 1
'''
religion_count = data.groupby('religion').count().shape

'''
answer 2
'''
data_wn = data.dropna()

'''
answer 3
'''
data_mod = data.copy()
data_mod['nevermarr'] = [0 if m == m else 1 for m in data_mod['agefm']]
data_mod = data_mod.drop('evermarr', axis = 1)
data_mod['agefm'] = [0 if m != m else m for m in data_mod['agefm']]
data_mod['heduc'] = [-1 if m['heduc'] != m['heduc'] and m['nevermarr'] == 1 else m['heduc'] for i, m in data_mod.iterrows()]
heduc_null_count = len(data_mod['heduc']) - sum(data_mod['heduc'].value_counts())

'''
answer 4
'''
data_mod['idlnchld_noans'] = [0 if m == m else 1 for m in data_mod['idlnchld']]
data_mod['heduc_noans'] = [0 if m == m else 1 for m in data_mod['heduc']]
data_mod['usemeth_noans'] = [0 if m == m else 1 for m in data_mod['usemeth']]

data_mod['heduc'] = [-2 if m['heduc'] != m['heduc'] and m['heduc_noans'] == 1 else m['heduc'] for i, m in data_mod.iterrows()]
data_mod['idlnchld'] = [-1 if m['idlnchld'] != m['idlnchld'] and m['idlnchld_noans'] == 1 else m['idlnchld'] for i, m in data_mod.iterrows()]
data_mod['usemeth'] = [-1 if m['usemeth'] != m['usemeth'] and m['usemeth_noans'] == 1 else m['usemeth'] for i, m in data_mod.iterrows()]

data_mod = data_mod.dropna(axis = 0, subset = ['knowmeth', 'electric', 'radio', 'tv', 'bicycle'])

answer4 = data_mod.shape[0] * data_mod.shape[1]

'''
answer 5, 6
'''
m1 = smf.ols('ceb ~ heduc + urban + electric + radio + tv + bicycle +'\
                    'nevermarr + idlnchld_noans + heduc_noans + usemeth_noans +'\
                    'age + educ + religion + idlnchld + knowmeth + usemeth +'\
                    'agefm', data=data_mod)
fitted = m1.fit()

'''
answer 7
'''
bp = sms.het_breushpagan(fitted.resid, fitted.model.exog)[1]

m2 = smf.ols('ceb ~ heduc + urban + electric + radio + tv + bicycle +'\
                    'nevermarr + idlnchld_noans + heduc_noans + usemeth_noans +'\
                    'age + educ + religion + idlnchld + knowmeth + usemeth +'\
                    'agefm', data=data_mod)
fitted2 = m2.fit(cov_type='HC1')

'''
answer 8
'''

m3 = smf.ols('ceb ~ heduc + urban + electric + bicycle +'\
                    'nevermarr + idlnchld_noans + heduc_noans + usemeth_noans +'\
                    'age + educ + idlnchld + knowmeth + usemeth +'\
                    'agefm', data=data_mod)
fitted3 = m3.fit()

m4 = smf.ols('ceb ~ heduc + urban + electric + bicycle +'\
                    'nevermarr + idlnchld_noans + heduc_noans + usemeth_noans +'\
                    'age + educ + idlnchld + knowmeth + usemeth +'\
                    'agefm', data=data_mod)
fitted4 = m4.fit(cov_type='HC1')

fisher = m2.fit().compare_f_test(m4.fit())

'''
answer 9
'''
m5 = smf.ols('ceb ~ heduc + urban + electric + bicycle +'\
                    'nevermarr + idlnchld_noans + heduc_noans + '\
                    'age + educ + idlnchld + knowmeth + '\
                    'agefm', data=data_mod)
fitted5 = m5.fit(cov_type='HC1')

fisher2 = m4.fit().compare_f_test(m5.fit())