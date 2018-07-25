# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 09:16:30 2018

@author: kazantseva
"""

import numpy as np
import matplotlib.pyplot as plt

plt.axhline(0, color='black')
plt.axvline(0, color='black')

x = np.arange(- 1, 1, 0.01)
y = -(x ** 7) + 5 * (x ** 3) - 3 * x
plt.plot(x, y)

y = - 7 * x ** 6 + 15 * x ** 2 - 3
plt.plot(x, y)

y = - 42 * x ** 5 + 30 * x
plt.plot(x, y)

'''
y = [4] * len(x)
plt.plot(x, y)
'''