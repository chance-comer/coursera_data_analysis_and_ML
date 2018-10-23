# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 15:04:32 2018

@author: kazantseva
"""

import scipy.stats as st
import pandas as pd
from sklearn import model_selection, linear_model, ensemble, metrics, cross_validation
from statsmodels.stats.weightstats import *

data = pd.read_csv('diamonds.txt', sep = '\t')

m_gen = 9.5
sigma = 0.4

m_sample = 9.57
sample_size = 160

z = (9.57 - 9.5) / (sigma / sample_size**0.5)
lvl = 2* (1 - st.norm.cdf(abs(z)))

train_sample, test_sample = cross_validation.train_test_split(data, test_size = 0.25, random_state = 1)

train_y = train_sample['price']
train_X = train_sample.drop(['price'], axis = 1)
test_y = test_sample['price']
test_X = test_sample.drop(['price'], axis = 1)

lr_model = linear_model.LinearRegression()
lr_model.fit(train_X, train_y)
lr_preds = lr_model.predict(test_X)
lr_score = metrics.mean_absolute_error(test_y, lr_preds)
lr_dev = abs(lr_preds - test_y)

rf_model = ensemble.RandomForestRegressor(random_state = 1)
rf_model.fit(train_X, train_y)
rf_preds = rf_model.predict(test_X)
rf_score = metrics.mean_absolute_error(test_y, rf_preds)
rf_dev = abs(rf_preds - test_y)

t_val = st.ttest_rel(lr_dev, rf_dev)

diff = lr_score - rf_score

cm = CompareMeans(DescrStatsW(lr_dev), DescrStatsW(rf_dev))