# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 09:55:01 2018

@author: kazantseva
"""

import scipy.stats as st
import pandas as pd
import numpy as np
from sklearn import model_selection, linear_model, metrics

'''
answer 3
'''
alpha = 0.05

n1 = 34
n2 = 16

success1 = 10
success2 = 4

p1 = success1 / n1
p2 = success2 / n2

P = (p1 * n1 + p2 * n2) / (n1 + n2)

z_3 = (p1 - p2) / (P * (1 - P) * (1 / n1 + 1 / n2)) ** 0.5

lvl_3 = 1 - st.norm.cdf(z_3)

'''
answer 4
'''

data = pd.read_csv('banknotes.txt', sep = '\t')

train_data, test_data = model_selection.train_test_split(data, test_size = 50, random_state = 1)

y = train_data['real']

data_1 = train_data[['X1', 'X2', 'X3']]
lr_1 = linear_model.LogisticRegression()
lr_1.fit(data_1, y)
preds_lr1 = lr_1.predict(test_data[['X1', 'X2', 'X3']])
score_lr1 = metrics.accuracy_score(preds_lr1, test_data['real'])
p_lr1 = 1 - score_lr1
lr1_iserror = [0 if t[0] == t[1] else 1 for t in zip(preds_lr1, test_data['real'])]

data_2 = train_data[['X4', 'X5', 'X6']]
lr_2 = linear_model.LogisticRegression()
lr_2.fit(data_2, y)
preds_lr2 = lr_2.predict(test_data[['X4', 'X5', 'X6']])
score_lr2 = metrics.accuracy_score(preds_lr2, test_data['real'])
p_lr2 = 1 - score_lr2
lr2_iserror = [0 if t[0] == t[1] else 1 for t in zip(preds_lr2, test_data['real'])]

f = sum([1 if t[0] == 1 and t[1] == 0 else 0 for t in zip(lr1_iserror, lr2_iserror)])
g =sum([1 if t[0] == 0 and t[1] == 1 else 0 for t in zip(lr1_iserror, lr2_iserror)])

z_4 = (f - g) / (f + g - (f - g)**2 / len(preds_lr1)) ** 0.5
lvl_4 = 2 * (1 - st.norm.cdf(abs(z_4)))

'''
answer 5
'''
z_5 = st.norm.ppf(1 - alpha / 2)

left_boundary = float(f - g) / len(preds_lr1)  - z_5 * np.sqrt(float((f + g)) / len(preds_lr1)**2 - float((f - g)**2) / len(preds_lr1)**3)
right_boundary = float(f - g) / len(preds_lr1)  + z_5 * np.sqrt(float((f + g)) / len(preds_lr1)**2 - float((f - g)**2) / len(preds_lr1)**3)

'''
answer 6
'''

n_6 = 200000
avg_6 = 525
sigma_6 = 100

sample_n = 100
sample_avg = 541.4

z_6 = (sample_avg - avg_6) / (sigma_6 / sample_n**0.5)
lvl_6 = 1 - st.norm.cdf(abs(z_6))

'''
answer 7
'''

sample_avg7 = 541.5

z_7 = (sample_avg7 - avg_6) / (sigma_6 / sample_n**0.5)
lvl_7 = 1 - st.norm.cdf(abs(z_7))
