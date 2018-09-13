# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 17:03:27 2018

@author: kazantseva
"""
from sklearn import datasets, cross_validation
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

digits = datasets.load_digits()
digits_x = digits.data
digits_y = digits.target

breast_cancer = datasets.load_breast_cancer()
breast_cancer_x = breast_cancer.data
breast_cancer_y = breast_cancer.target

gnb = GaussianNB()
est1 = cross_validation.cross_val_score(gnb, digits_x, digits_y).mean()
est2 = cross_validation.cross_val_score(gnb, breast_cancer_x, breast_cancer_y).mean()

print(est1, est2)

bnb = BernoulliNB()
est1 = cross_validation.cross_val_score(bnb, digits_x, digits_y).mean()
est2 = cross_validation.cross_val_score(bnb, breast_cancer_x, breast_cancer_y).mean()

print(est1, est2)

mnb = MultinomialNB()
est1 = cross_validation.cross_val_score(mnb, digits_x, digits_y).mean()
est2 = cross_validation.cross_val_score(mnb, breast_cancer_x, breast_cancer_y).mean()

print(est1, est2)