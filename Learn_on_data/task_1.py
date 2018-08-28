# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 17:36:33 2018

@author: kazantseva
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("weights_heights.csv", index_col = 'Index')
#data.plot(y = 'Height', title = 'Height inch', color = 'blue', kind = 'hist' )

def make_bmi(weight_pound, height_inch):
    kg_to_pound, METER_TO_INCH =  2.20462, 39.37
    return (weight_pound / kg_to_pound) / (height_inch / METER_TO_INCH) ** 2

data['BMI'] = data.apply(lambda row: make_bmi(row['Weight'], row['Height']), axis = 1)

#sns.pairplot(data)

data.plot(x = 'Weight', y = 'Height', kind = 'scatter', title = 'Зависимость роста от веса')

#plt.scatter(x = data['Weight'], y = data['Height'])