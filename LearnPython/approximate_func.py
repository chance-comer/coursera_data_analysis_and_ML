# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 16:38:58 2018

@author: kazantseva
"""

import scipy.linalg as sl
import numpy as np
import matplotlib.pyplot as plt

def valfunc(x):
  return np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(- x / 2)

x = np.arange(0, 16, 0.1)

plt.plot(x, valfunc(x))

res = sl.solve([[1, 1], [1, 15]], [valfunc(1), valfunc(15)])

def appfunc(x):
  return res[0] + res[1] * x

plt.plot(x, appfunc(x))

res = sl.solve([[1, 1, 1, 1], [1, 4, 16, 64], [1, 10, 100, 1000], [1, 15, 225, 3375]], [valfunc(1), valfunc(4), valfunc(10), valfunc(15)])

def appfunc2(x):
  return res[0] + res[1] * x + res[2] * x ** 2 + res[3] * x ** 3

plt.plot(x, appfunc2(x))