# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 17:36:33 2018

@author: kazantseva
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.optimize as opt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv("weights_heights.csv", index_col = 'Index')
#data.plot(y = 'Height', title = 'Height inch', color = 'blue', kind = 'hist' )

def make_bmi(weight_pound, height_inch):
    kg_to_pound, METER_TO_INCH =  2.20462, 39.37
    return (weight_pound / kg_to_pound) / (height_inch / METER_TO_INCH) ** 2

data['BMI'] = data.apply(lambda row: make_bmi(row['Weight'], row['Height']), axis = 1)

#sns.pairplot(data)

f = data.plot(x = 'Weight', y = 'Height', kind = 'scatter', title = 'Зависимость роста от веса')

#plt.scatter(x = data['Weight'], y = data['Height'])
def compute_error(w):  
  y_hat = w[0] + w[1] * data['Weight']
  error = (data['Height'] - y_hat) ** 2
  return error.sum()

#print(compute_error(1, 1))
x = np.linspace(70, 170, len(data))
plt.scatter(data['Weight'], data['Height'])
plt.plot(x, 60 + 0.05 * x, color = 'orange')
plt.plot(x, 50 + 0.16 * x, color = 'violet')
plt.xlabel("Вес")
plt.ylabel("Рост")
plt.title("Случайные регрессионные прямые")
plt.show()

w1 = np.linspace(-5, 5, len(data))
#error = [compute_error(50, w) for w in w1]
#plt.plot(w1, error)
res = opt.minimize_scalar(lambda x: compute_error([50, x]), bounds=(-5,5))

fig = plt.figure()
ax = fig.gca(projection='3d') # get current axis

# Создаем массивы NumPy с координатами точек по осям X и У. 
# Используем метод meshgrid, при котором по векторам координат 
# создается матрица координат. Задаем нужную функцию Z(x, y).
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
Z = [[compute_error([w0, w1]) for w0 in X] for w1 in Y]
X, Y = np.meshgrid(X, Y)
# Наконец, используем метод *plot_surface* объекта 
# типа Axes3DSubplot. Также подписываем оси.
surf = ax.plot_surface(X, Y, Z)
ax.set_xlabel('Intercept')
ax.set_ylabel('Slope')
ax.set_zlabel('Error')
plt.show()

answer = opt.minimize(compute_error, x0 = [0, 0], method = 'L-BFGS-B', bounds = [(-100, 100), (-5, 5)])