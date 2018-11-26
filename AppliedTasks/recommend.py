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

sorted_view = freq_table.sort_values('view_freq', ascending = False)
#top_view_1 = sorted_view[:1]
#top_view_5 = sorted_view[:5]

sorted_bought = freq_table.sort_values('bought_freq', ascending = False)
#top_bought_1 = sorted_bought[:1]
#tope_bought_5 = sorted_bought[:5]

def avg_pr_rec(file_name, recommends_count = 5, is_topView = True):
  with open(file_name) as f:
    something_bought = 0
    precision = 0
    recall = 0
  
    for line in f:
      tr_line = line.strip()
      ar_v_b = tr_line.split(';')    
      if ar_v_b[1] != '':
        something_bought += 1
      
        b_str = ar_v_b[1].split(',')
        v_str = ar_v_b[0].split(',')
        b = [int(a) for a in b_str]
        v = [int(a) for a in v_str]
        v_len = len(v)
        
        viewed_freq = freq_table[freq_table['product_id'].isin(v)]
        viewed_freq['num'] = 0
        
        for i,val in enumerate(v):
          if viewed_freq[viewed_freq['product_id'] == i] is not None:
            #viewed_freq[viewed_freq['product_id'] == i]['num'] = i
            viewed_freq.loc[viewed_freq['product_id'] == i, 'num'] = i
          else:
            viewed_freq.append([val, 0, 0, i])
        
        real_recommends_count = min(recommends_count, v_len)
        real_recommends = sorted_view[:real_recommends_count]['product_id'] if is_topView else sorted_bought[:real_recommends_count]['product_id']
        from_recommends_bought = len(set(real_recommends) & set(b))
        precision += from_recommends_bought / recommends_count
        recall += from_recommends_bought / len(b)
    avg_precision = precision / something_bought
    avg_recall = recall / something_bought
    return avg_precision, avg_recall

train_avg_precision_1rec_view, train_avg_recall_1rec_view = avg_pr_rec('coursera_sessions_train.txt', recommends_count = 1)
train_avg_precision_1rec_bought, train_avg_recall_1rec_bought = avg_pr_rec('coursera_sessions_train.txt', recommends_count = 1, is_topView = False)
train_avg_precision_5rec_view, train_avg_recall_5rec_view = avg_pr_rec('coursera_sessions_train.txt')
train_avg_precision_5rec_bought, train_avg_recall_5rec_bought = avg_pr_rec('coursera_sessions_train.txt', is_topView = False)

test_avg_precision_1rec_view, test_avg_recall_1rec_view = avg_pr_rec('coursera_sessions_test.txt', recommends_count = 1)
test_avg_precision_1rec_bought, test_avg_recall_1rec_bought = avg_pr_rec('coursera_sessions_test.txt', recommends_count = 1, is_topView = False)
test_avg_precision_5rec_view, test_avg_recall_5rec_view = avg_pr_rec('coursera_sessions_test.txt')
test_avg_precision_5rec_bought, test_avg_recall_5rec_bought = avg_pr_rec('coursera_sessions_test.txt', is_topView = False)

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