# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 17:20:11 2018

@author: kazantseva
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
#from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import cross_val_score
#from sklearn.metrics import f1_score
from sklearn.cross_validation import StratifiedKFold

data = pd.read_csv('SMSSpamCollection.txt', sep = '\t', names = ['class', 'text'], header = None)

y = np.where(data['class'] == 'spam', 1, 0)

vectorizer = CountVectorizer()
vectorizer.fit(data['text'])
X = vectorizer.transform(data['text']) 

log_model = LogisticRegression()
cv = StratifiedKFold(y, n_folds = 10, random_state = 2)
log_score = cross_val_score(log_model, X, y, cv = cv, scoring = 'f1')

mean_log_score = log_score.mean() # 0.9326402983610631

log_model_whole = LogisticRegression()
log_model_whole.fit(X, y)
input_1 = vectorizer.transform(['FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! Subscribe6GB'])
pred_1 = log_model_whole.predict(input_1) # 1
input_2 = vectorizer.transform(['FreeMsg: Txt: claim your reward of 3 hours talk time'])
pred_2 = log_model_whole.predict(input_2) # 1
input_3 = vectorizer.transform(['Have you visited the last lecture on physics?'])
pred_3 = log_model_whole.predict(input_3) # 0
input_4 = vectorizer.transform(['Have you visited the last lecture on physics? Just buy this book and you will have all materials! Only 99$'])
pred_4 = log_model_whole.predict(input_4) # 0
input_5 = vectorizer.transform(['Only 99$'])
pred_5 = log_model_whole.predict(input_5) # 0

bigram_vectorizer = CountVectorizer(ngram_range = (2, 2))
bigram_vectorizer.fit(data['text'])
bigram_X = bigram_vectorizer.transform(data['text'])
bigram_log_score = cross_val_score(log_model, bigram_X, y, cv = cv, scoring = 'f1')
bigram_mean_log_score = bigram_log_score.mean() # 0.82242206641871329

trigram_vectorizer = CountVectorizer(ngram_range = (3, 3))
trigram_vectorizer.fit(data['text'])
trigram_X = trigram_vectorizer.transform(data['text'])
trigram_log_score = cross_val_score(log_model, trigram_X, y, cv = cv, scoring = 'f1')
trigram_mean_log_score = trigram_log_score.mean() # 0.72501615554673771

unigram_vectorizer = CountVectorizer(ngram_range = (1, 3))
unigram_vectorizer.fit(data['text'])
unigram_X = unigram_vectorizer.transform(data['text'])
unigram_log_score = cross_val_score(log_model, unigram_X, y, cv = cv, scoring = 'f1')
unigram_mean_log_score = unigram_log_score.mean() # 0.92513825586488374

bias_model = MultinomialNB()

bigram_bias_score = cross_val_score(bias_model, bigram_X, y, cv = cv, scoring = 'f1')
bigram_mean_bias_score = bigram_bias_score.mean() # 0.64550151779854426

trigram_bias_score = cross_val_score(bias_model, trigram_X, y, cv = cv, scoring = 'f1')
trigram_mean_bias_score = trigram_bias_score.mean() # 0.37871948524573595

unigram_bias_score = cross_val_score(bias_model, unigram_X, y, cv = cv, scoring = 'f1')
unigram_mean_bias_score = unigram_bias_score.mean() # 0.88848596560610016

tfidf_vectorizer = TfidfVectorizer(ngram_range = (1, 1))
tfidf_vectorizer.fit(data['text'])
tfidf_X = tfidf_vectorizer.transform(data['text'])
tf_idf_log_score = cross_val_score(log_model, tfidf_X, y, cv = cv, scoring = 'f1')

mean_tfidf_log_score = tf_idf_log_score.mean() # 0.85285995541724557
