# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 16:43:01 2018

@author: kazantseva
"""
import re
import numpy as np
import pandas as pd
import scipy.spatial.distance as sd

text = []

# read lines from file
with open('sentences.txt', mode = 'r') as file_obj:
  text = file_obj.readlines()

#make lower case
text = [ls.lower() for ls in text]

#split words in every sentence 
#([^a-zA-Z\'0-9] more appropriate but in task said we need matrix with 254 columns)
words_in_sentences = [re.split('[^a-z]', ls) for ls in text]

#filter empty words
words_in_sentences = [list(filter(lambda w : w != '', ws)) for ws in words_in_sentences]

#get unique words (flatten list then make set from list to delete duplicates)
words = [ word for words_in_sentence in words_in_sentences for word in words_in_sentence ]
unique_words = set(words)


word_count_matrix = []

#get count of each word in each sentence
for words_in_sentence in words_in_sentences:
  words_count_dict = {}
  for word in unique_words:
    words_count_dict[word] = words_in_sentence.count(word)
  word_count_matrix.append(words_count_dict)
  
#data = pd.DataFrame(word_count_matrix)

#test = pd.DataFrame([{'d': 1, 'f': 2, 'c': 3}, {'d' :4, 'c': 5, 'f' : 6}])
  
distance_matrix = [[sd.cosine(list(word_count_i.values()), list(word_count_j.values())) for i, word_count_i in enumerate(word_count_matrix)] for j, word_count_j in enumerate(word_count_matrix)]

first_distance = pd.DataFrame(distance_matrix[0], columns = ['dist'])
first_distance = first_distance.drop([0])
first_distance_sorted = first_distance.sort_values('dist')

