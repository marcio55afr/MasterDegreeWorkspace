# -*- coding: utf-8 -*-


import pandas as pd

variant = "V1_clf/"
test = 'ST_'+variant
file = 'results/'+test+'results.csv'

df = pd.read_csv(file)
print(df.iloc[:,[0,-5,-1]])
