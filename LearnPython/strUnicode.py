# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 11:46:10 2018

@author: kazantseva
"""
import pandas as pd
import numpy as np
import scipy as sp
from scipy import optimize as opt
from scipy import linalg as la
import matplotlib.pyplot as plt
from scipy import interpolate as ip

a = ['высокий', 'средний']
file_obj = open('tt1.html', mode = 'a')
file_obj.write('g\n')
file_obj.close()

data = pd.read_csv('iris.csv')

data = data[:5]
data['custom'] = 'q', 'w', 'e', 'r', 't'

ar = [1, 2, 3, 4, 5]
npar = np.array(ar)
randar = np.random.rand(4, 4)

def f(x):
  return (x[0] - 3.2) ** 2 + (x[1] - 0.1) ** 2 + 3

x_min = opt.minimize(f, [5, 5])

x1 = [[1, 4, 6], [5, 1, 3], [7, 3, 4]] 
x2 = [5, 8, 2]

x3 = la.solve(x1, x2)
x4 = np.dot(x1, x3)

arg = np.arange(0, 10, 2)
f = np.exp(-arg / 3.0)

#plt.plot(arg, f)

approximate = ip.interp1d(arg, f, kind = 'quadratic')
xnew = np.arange(0, 8, 0.1)
ynew = approximate(xnew)

plt.plot(arg, f, 'o', xnew, ynew, '-')

