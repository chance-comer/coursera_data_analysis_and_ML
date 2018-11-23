# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 11:36:59 2018

@author: kazantseva
"""

import pandas as pd
import collections as c
import numpy as np

viewed = np.array([])
bought = np.array([])
'''
with open('coursera_sessions_train.txt') as f:
  l = f.readline().strip()
  ar_v_b = l.split(';')
  print(ar_v_b[1] is '')
'''
f = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns = ['one', 'two', 'three'])
f.to_csv('test.txt', index = False)

'''
with open('coursera_sessions_train.txt') as f:
  for line in f:
    tr_line = line.strip()
    ar_v_b = tr_line.split(';') 
    v = ar_v_b[0].split(',')
    viewed = np.append(viewed, v)
    if ar_v_b[1] is not '':      
      b = ar_v_b[1].split(',')      
      bought = np.append(bought, b)

bought_counter = c.Counter(bought)
viewed_counter = c.Counter(viewed)

freq_table = pd.DataFrame(data = [[k, viewed_counter[k], bought_counter[k]] for k in viewed_counter.keys()], columns = ['session_id', 'view_freq', 'bought_freq'])
freq_table.to_csv('freq_table.txt', index = False)
'''
freq_table = pd.read_csv('freq_table.txt')

viewed_test = np.array([])
bought_test = np.array([])
'''
with open('coursera_sessions_test.txt') as f:
  for line in f:
    tr_line = line.strip()
    ar_v_b = tr_line.split(';') 
    v = ar_v_b[0].split(',')
    viewed_test = np.append(viewed_test, v)
    if ar_v_b[1] is not '':      
      b = ar_v_b[1].split(',')      
      bought_test = np.append(bought_test, b)
      
bought_test_counter = c.Counter(bought_test)
viewed_test_counter = c.Counter(viewed_test)
'''