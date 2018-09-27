# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 09:43:08 2018

@author: kazantseva
"""

import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift

data = pd.read_csv('checkins.dat', sep = '|', skiprows = 2, header = None)
header = pd.read_csv('checkins.dat', sep = '|', nrows = 0)
header.rename(columns = lambda x : x.strip(), inplace = True)
data.replace(' ' * 19, np.nan, inplace = True)
data.dropna(inplace = True)
data.columns = header.columns
#data.strip()

subset = data[['latitude', 'longitude']][:100000]

subset['latitude'] = subset['latitude'].astype(float)
subset['longitude'] = subset['longitude'].astype(float)

clusterator = MeanShift(bandwidth=0.1)
clusterator.fit(subset)

centers = clusterator.cluster_centers_
lbls = clusterator.labels_
cluster_size = [(lbl_centr, len(list(filter(lambda x: x == lbl_centr[0], lbls)))) for lbl_centr in zip(np.unique(lbls), centers)]
most_popular = list(filter(lambda x: x[1] > 15, cluster_size))
ind_for_dist = [mp[0][0] for mp in most_popular]

offices = { '(Los Angeles)' : [33.751277, -118.188740],\
            '(Miami)' : [25.867736, -80.324116 ], \
            '(London)' : [51.503016, -0.075479 ],\
            '(Amsterdam)' : [52.378894, 4.885084], \
            '(Beijing)' : [39.366487, 117.036146],\
            '(Sydney)' : [-33.868457, 151.205134] }

distances = pd.DataFrame(columns = ['(Los Angeles)', '(Miami)', '(London)', '(Amsterdam)', '(Beijing)', '(Sydney)'])

for mp in most_popular:
  d = {}
  for k_office, v_office in offices.items():
    d[k_office] = np.linalg.norm(np.array(mp[0][1]) - np.array(v_office))
  distances.loc[mp[0][0]] = d
  
mindist = distances.min(axis = 1).sort_values()[:20]