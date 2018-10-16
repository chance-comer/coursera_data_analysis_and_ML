# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 15:32:25 2018

@author: kazantseva
"""

import pandas as pd
from statsmodels.stats.weightstats import _zconfint_generic, _tconfint_generic

data = pd.read_csv('water.txt', sep = '\t')

mean_mortality = data['mortality'].mean()
std_mortality = data['mortality'].std(ddof = 1)

interval = _tconfint_generic(mean_mortality, std_mortality/len(data)**0.5, len(data) - 1, 0.05, 'two-sided')

mean_mortality_s = data[data['location'] == 'South']['mortality'].mean()
std_mortality_s = data[data['location'] == 'South']['mortality'].std(ddof = 1)

interval_s = _tconfint_generic(mean_mortality_s, std_mortality_s/len(data[data['location'] == 'South'])**0.5, len(data[data['location'] == 'South']) - 1, 0.05, 'two-sided')

mean_mortality_n = data[data['location'] == 'North']['mortality'].mean()
std_mortality_n = data[data['location'] == 'North']['mortality'].std(ddof = 1)

interval_n = _tconfint_generic(mean_mortality_n, std_mortality_n/len(data[data['location'] == 'North'])**0.5, len(data[data['location'] == 'North']) - 1, 0.05, 'two-sided')

mean_hardness_s = data[data['location'] == 'South']['hardness'].mean()
std_hardness_s = data[data['location'] == 'South']['hardness'].std(ddof = 1)

interval_s_h = _tconfint_generic(mean_hardness_s, std_hardness_s/len(data[data['location'] == 'South'])**0.5, len(data[data['location'] == 'South']) - 1, 0.05, 'two-sided')

mean_hardness_n = data[data['location'] == 'North']['hardness'].mean()
std_hardness_n = data[data['location'] == 'North']['hardness'].std(ddof = 1)

interval_n_h = _tconfint_generic(mean_hardness_n, std_hardness_n/len(data[data['location'] == 'North'])**0.5, len(data[data['location'] == 'North']) - 1, 0.05, 'two-sided')