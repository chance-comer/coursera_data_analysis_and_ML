# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 15:27:03 2018

@author: kazantseva
"""

import pandas as pd

df = pd.read_csv('bikes_rent.csv')

df.iloc[:,:-1].corrwith(df['cnt'])
df[['temp', 'atemp', 'hum', 'windspeed(mph)', 'windspeed(ms)', 'cnt']].corr()

from sklearn.preprocessing import StandardScaler

data = [[0, 0], [0, 0], [1, 1], [1, 1]]
scaler = StandardScaler()
print(scaler.fit(data))

print(scaler.mean_)

print(scaler.transform(data))

print(scaler.transform([[2, 2]]))
