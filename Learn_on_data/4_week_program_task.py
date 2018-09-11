# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 16:42:05 2018

@author: kazantseva
"""

import sklearn.datasets as ds
impoer numpy as np
import pandas as pd
import sklearn.cross_validation as cross_val
import sklearn.tree as tree
import sklearn.metrics as metrics
import sklearn.model_selection as ms
import sklearn.ensemble as ensemble
import matplotlib.pyplot as plt

data = ds.load_digits()

X = data.data
y = data.target

clf = tree.DecisionTreeClassifier()
quality = ms.cross_val_score(clf, X, y, cv = 10)
answer_1 = quality.mean()

clf_2 = ensemble.BaggingClassifier(n_estimators = 100)
quality_2 = ms.cross_val_score(clf_2, X, y, cv = 10)
answer_2 = quality_2.mean() 

clf_3 = ensemble.BaggingClassifier(n_estimators = 100, max_features = np.sqrt(len(data.data[0])))
quality_3 = ms.cross_val_score(clf_3, X, y, cv = 10)
answer_3 = quality_3.mean() 

dt = tree.DecisionTreeClassifier(max_features = np.sqrt(len(data.data[0])))
clf_4 = ensemble.BaggingClassifier(dt, n_estimators = 100)
quality_4 = ms.cross_val_score(clf_3, X, y, cv = 10)
answer_4 = quality_4.mean() 

clf_5 = ensemble.RandomForestClassifier(n_estimators = 100, max_features = np.sqrt(len(data.data[0])))
quality_5 = ms.cross_val_score(clf_5, X, y, cv = 10)

trees_count = [1, 5, 10, 50, 100]
max_features = [1, 5, 15, 50, 64]
max_depth = [1, 3, 5, 10, 15]

tc_quality = []
for tc in trees_count:
  clf_tc = ensemble.RandomForestClassifier(n_estimators = 100, max_features = np.sqrt(len(data.data[0])))
  tc_quality.append(ms.cross_val_score(clf_tc, X, y, cv = 10).mean())

plt.plot(trees_count, tc_quality)
  