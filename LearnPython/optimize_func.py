# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 12:17:12 2018

@author: kazantseva
"""
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

def valfunc(x):
  return np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(- x / 2)

def valintfunc(x):
  return np.array([int(i) for i in valfunc(x)])

res = opt.minimize(valfunc, [30], method = 'BFGS')
res1 = opt.differential_evolution(valfunc, [(1, 30)])

x = np.arange(1, 30, 0.1)
y = valintfunc(x)

res2 = opt.minimize(valintfunc, [30], method = 'BFGS')
res3 = opt.differential_evolution(valintfunc, [(1, 30)])

plt.plot(x, y)