# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 11:58:20 2018

@author: kazantseva
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

data = pd.read_csv('monthly-milk-production.csv', sep = ';', index_col=['month'], \
                   parse_dates=['month'], dayfirst=True)

diki_fuller = sm.tsa.stattools.adfuller(data['milk'])

data_div = pd.DataFrame(list(map(lambda x: x[1] / x[0].days_in_month, data.iterrows())))

answer_5 = sum(data_div['milk'])

data_div.daily_diff1 = data_div.milk - data_div.milk.shift(12)
diki_fuller_1 = sm.tsa.stattools.adfuller(data_div.daily_diff1.dropna())
data_div.daily_diff2 = data_div.daily_diff1 - data_div.daily_diff1.shift(1)
diki_fuller_2 = sm.tsa.stattools.adfuller(data_div.daily_diff2.dropna())

sm.graphics.tsa.plot_acf(data_div.daily_diff2.dropna().values.squeeze(), lags=50)
sm.graphics.tsa.plot_pacf(data_div.daily_diff2.dropna().values.squeeze(), lags=50)