# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 17:01:02 2018

@author: kazantseva
"""

import pandas as pd
import numpy as np

adver_data = pd.read_csv("advertising.csv")

X = np.array([adver_data["TV"].values, adver_data["Radio"].values, adver_data["Newspaper"].values]) # Ваш код здесь
y = np.array(adver_data["Sales"].values) # Ваш код здесь

means, stds = X.mean(axis = 1), X.std(axis = 1)# Ваш код здесь

X = (X - np.repeat(means, len(adver_data), axis = 0).reshape(3, len(adver_data))) / \
    np.repeat(stds, len(adver_data), axis = 0).reshape(3, len(adver_data))

X = X.T
X = np.hstack((np.ones(len(X)).reshape(len(X), 1), X))

#a = [[1, 2, 3], [4, 5, 6]]
#o = np.ones(len(a)).reshape(len(a), 1)

#b = np.hstack((o, a))
y.