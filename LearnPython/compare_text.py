# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 16:43:01 2018

@author: kazantseva
"""
import re
import numpy as np

text = []

# read lines from file
with open('sentences.txt', mode = 'r') as file_obj:
  text = file_obj.readlines()

#make lower case
text = [ls.lower() for ls in text]

#split words in every sentence
words = [re.split('[^a-zA-Z\'0-9]', ls) for ls in text]

#filter empty words
words = [list(filter(lambda w : w != '', ws)) for ws in words]

a = [[1, 2],[1],[1, 4, 5, 8],[11, 15, 16],[20]]
b = {2, 3}
b.add(3)

unique_words = np.unique(np.array(words).reshape(-1))