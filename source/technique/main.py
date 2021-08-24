

import pandas as pd
import numpy as np
from sktime.classification.base import BaseClassifier
from sklearn.feature_extraction import DictVectorizer

from source.utils import ResolutionMatrix, NgramExtractor
from source.transformations import MSAX
from source.technique import WordRanking



class SearchTechnique(BaseClassifier):
    
    
    
    def __init__(self,
                 ts_length,
                 alphabet_size = 4,
                 word_length = 6,
                 num_windows = None,
                 max_window_length = None):
        
        
        self.ts_length = ts_length
        self.alphabet_size = alphabet_size
        self.word_length = word_length
        self.num_windows = num_windows
        self.max_window_length = max_window_length
        
        
        self.resolution_matrix = ResolutionMatrix(ts_length,
                                                  word_length,
                                                  num_windows,
                                                  max_window_length)
        self.discretizer = MSAX(alphabet_size)   
    
    
    def fit(self, data, labels):
        
        if( data.shape[0] != labels.shape[0] ):
            raise RuntimeError('The labels isn\'t compatible with the data received')
            
        data = data.squeeze()                
        bob = self._extract_bob_from(data, labels)
        
        
        WHERE DO I PUT THIS VECTORIZATION????
        
        #vectorizer = DictVectorizer(sparse=True, dtype=np.int32, sort=False)
        #bag_vec = vectorizer.fit_transform(bob)
        
        #WordRanking.chi2(bag_vec, labels)
    
    def _extract_bob_from(self, timeseries_set, labels):
        
        rm = self.resolution_matrix
        windows =  rm.get_windows()
        
        i=0
        n_samples = timeseries_set.shape[0]
        bag_of_bags = pd.DataFrame()
        print("Generating the bag of bags")
        print("__________")
        for ts in timeseries_set:
            if(i%(n_samples/10) == 0):
                print('#',end='')
            word_seq = self.discretizer.transform(ts, windows)
            bob = NgramExtractor.get_bob(word_seq, rm.matrix)
            bob['sample'] = i
            i+=1
            bag_of_bags = pd.concat([bag_of_bags,bob], axis=0, join='outer', ignore_index=True)
        
        
        
        bag_of_bags['label'] = labels.loc[bag_of_bags['sample']].values
        
        return bag_of_bags
