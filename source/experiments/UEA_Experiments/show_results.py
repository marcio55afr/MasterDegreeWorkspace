# -*- coding: utf-8 -*-


import pandas as pd

variant = "V2/"
test = 'ST_'+variant
file = 'results/'+test+'results.csv'

df = pd.read_csv(file)
df = df.sort_values('ROC AUC mean')
print(df.iloc[:,[0,-5,-1]])
print('\n\n')
df = df.sort_values('ROC AUC efficency')
print(df.iloc[:,[0,-5,-1]])
