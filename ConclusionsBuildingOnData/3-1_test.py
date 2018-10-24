# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 10:04:51 2018

@author: kazantseva
"""

import pandas as pd

data = pd.read_csv('illiteracy.txt', sep = '\t')

answer_4 = data.corr()['Illit']['Births']

answer_5 = data.corr(method = 'spearman')['Illit']['Births'] #0.75296221373253402 answer as 0.7530