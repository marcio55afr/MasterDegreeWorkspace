# -*- coding: utf-8 -*-


import pandas as pd

variant = "V4/"
test = 'ST_'+variant
file = 'results_windows/'+test+'results.csv'

df = pd.read_csv(file)
df = df.set_index('strategy_name')
df = df.sort_values('ROC AUC mean')
#df = df.sort_values('AUROC mean')
print(df.iloc[:,[3]])
print('\n\n')
print(df.iloc[:,[-5,-1]])
print('\n\n')
df = df.sort_values('Accuracy mean')
print(df.iloc[:,[-5,-1]])

df.to_csv(file)

print()
print(df.min())
print()
print(df.max())