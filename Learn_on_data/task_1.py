# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 17:36:33 2018

@author: kazantseva
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("weights_heights.csv")
data.plot(y = 'Height', title = 'Height inch', color = 'blue', kind = 'hist' )