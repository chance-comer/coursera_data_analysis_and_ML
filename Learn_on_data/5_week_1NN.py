# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 10:29:16 2018

@author: kazantseva
"""

from sklearn import datasets
from sklearn import ensemble
from sklearn import cross_validation
from sklearn import model_selection
import numpy as np
from sklearn import metrics

data = datasets.load_digits()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.25, shuffle = False)

def oneNN(test_obj, X_train):  
  min_dist = -1
  closer = -1
  for i in range(len(X_train)):
    dist = np.linalg.norm(X_train[i] - test_obj)
    if (min_dist < 0 or dist < min_dist):
      min_dist = dist
      closer = i
  return closer
  
prediction = []

for obj in X_test:
  prediction.append(y_train[oneNN(obj, X_train)])

accuracy = metrics.accuracy_score(prediction, y_test)

#accuracy_2 = -1
#success = 0
 
#for i, pr in enumerate(prediction):
  #if pr == y_test[i]:
    #success += 1

#accuracy_2 = success / len(y_test)

rf_clf = ensemble.RandomForestClassifier(n_estimators=1000).fit(X_train, y_train)
prediction_2 = rf_clf.predict(X_test)

accuracy_2 = metrics.accuracy_score(prediction_2, y_test)

