# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 15:27:03 2018

@author: kazantseva
"""

import pandas as pd

df = pd.read_csv('bikes_rent.csv')

df.iloc[:,:-1].corrwith(df['cnt'])