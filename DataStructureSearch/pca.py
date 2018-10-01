# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 14:29:01 2018

@author: kazantseva
"""

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_score as cv_score
import matplotlib.pyplot as plt

def plot_variances(d_variances):
    n_components = np.arange(1,d_variances.size+1)
    plt.plot(n_components, d_variances, 'b', label='Component variances')
    plt.xlim(n_components[0], n_components[-1])
    plt.xlabel('n components')
    plt.ylabel('variance')
    plt.legend(loc='upper right')
    plt.show()
    
def write_answer_2(optimal_d):
    with open("pca_answer2.txt", "w") as fout:
        fout.write(str(optimal_d))
        
data = pd.read_csv('data_task2.csv')

# place your code here
model = PCA(n_components = len(data.columns))
model.fit(data)
tranformed_data = model.transform(data)
variance = tranformed_data.std(axis = 0)
sorted_variance = sorted(variance, key = lambda x: -x)
diff = [sorted_variance[i - 1] - variance for i, variance in enumerate(sorted_variance) if i > 0]
max_diff = max(diff)
arg_diff_max = np.argmax(diff)
plot_variances(variance)
#printtranformed_data[0]
#print sorted_variance
#print diff
#print max_diff
write_answer_2(arg_diff_max + 1)