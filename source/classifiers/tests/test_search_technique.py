# -*- coding: utf-8 -*-
import sys
sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace/source')
sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace')

from source.classifiers import SearchTechnique
from source.experiments.database import get_dataset, get_train_test_split, DATASET_NAMES

from sklearn.metrics import accuracy_score
import pandas as pd
import unittest

class TestFunction_fit(unittest.TestCase):
    
    def __init__(self):
        self.data, self.labels = get_train_test_split(DATASET_NAMES[0], split='train')
        self.ts_length = self.data.iloc[0,0].size        
    
    def test_NoErrors(self):
        
        clf = SearchTechnique(self.ts_length)
        clf.fit(self.data, self.labels)
        
        

train, labels = get_train_test_split(DATASET_NAMES[0], split='train')
ts_length = train.iloc[0,0].size

clf = SearchTechnique(ts_length)
rm = clf.resolution_matrix.matrix.copy()

acc = pd.DataFrame()
for window in rm:
    for ngram in range(1,24):
        if(rm.loc[ngram,window] > 0):
            print('window: {}'.format(window))
            print('ngram: {}'.format(ngram))
            
            unique_resolution = pd.DataFrame(data=False, index=rm.index, columns=[window])
            unique_resolution.loc[ngram,window] = 1            
            
            clf = SearchTechnique(ts_length)
            clf.resolution_matrix.matrix = unique_resolution   
            clf.fit(train, labels)
            
            
            test, y_true = get_train_test_split(DATASET_NAMES[0], split='test')
            y_pred = clf.predict(test)
            
            acc.loc[ngram,window] = accuracy_score(y_true, y_pred)
acc = acc.fillna('')
print('acc')
print(acc)
