import sys
sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace/source')
sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace')

import time
import pandas as pd
import numpy as np
from sktime.classification.base import BaseClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from source.utils import ResolutionHandler, ResolutionMatrix, NgramExtractor
from source.transformations import MSAX
from source.technique.word_ranking import WordRanking
from source.technique.resolution_selector import ResolutionSelector


class SearchTechnique(BaseClassifier):
    
    
    
    def __init__(self,
                 ts_length,
                 word_ranking_method = 'chi2',
                 word_selection = 'best n words', # ['p threshold', 'best n words']
                 p_threshold = 0.05,
                 n_words = 120,
                 alphabet_size = 4,
                 word_length = 6,
                 num_windows = None,
                 max_window_length = None,
                 random_state = None):
        
        
        if (word_selection != 'p threshold') and (word_selection != 'best n words'):
            raise ValueError('The word selection must the a valid method of selection, as "p threshold" or "best n words"')
        
        
        self.ts_length = ts_length
        self.word_ranking_method = word_ranking_method
        self.word_selection = word_selection
        self.p_threshold = p_threshold
        self.n_words = n_words
        
        
        self.alphabet_size = alphabet_size
        self.word_length = word_length
        self.num_windows = num_windows
        self.max_window_length = max_window_length
        
        self.random_state = random_state
        self.resolution_matrix = ResolutionMatrix(ts_length,
                                                  word_length,
                                                  num_windows,
                                                  max_window_length)
        self.discretizer = MSAX(alphabet_size)
        
        self.clf = LogisticRegression(max_iter=500,
                                      random_state=random_state)
        
        self.selected_words = set()
    
    
    def fit(self, data, labels):
        
        if( data.shape[0] != labels.shape[0] ):
            raise RuntimeError('The labels isn\'t compatible with the data received')
        
        # testing one specific dataset to avoid rework
        folder_path = 'C:/Users/marci/Desktop/MasterDegreeWorkspace/source/experiments/data_visualizing/worms/WordLen_6/'
        bob_path = folder_path+'/bag_of_bags_train.csv'
        bob = pd.read_csv(bob_path)
        
        # testing one specific resolution
        #matrix = self.resolution_matrix.matrix
        #window = matrix.columns[0]
        #ngram = matrix[matrix[window] > 0].index[0]
        #bob = bob[ bob.window == window]
        #bob = bob[ bob.ngram == ngram]
                
        #data = data.squeeze()                
        #bob = self._extract_bob_from(data, labels)
        
        feature_vec = bob.pivot(index='sample', columns='ngram word', values='frequency')
        feature_vec = feature_vec.fillna(0)
        del(bob)

        word_rank = pd.DataFrame(index = feature_vec.columns.values)
        rank_value, p = WordRanking.get_ranking( self.word_ranking_method, feature_vec, labels )
        word_rank['rank'] = rank_value
        word_rank['p'] = p
        
        resolution = word_rank.index.map( lambda x: ResolutionHandler.get_resolution_from(x) )
        word_rank['resolution'] = resolution        
        best_reso = ResolutionSelector.get_best_resolution_max(word_rank)
        
        word_rank.index.name = 'ngram word'
        word_rank = word_rank.reset_index().set_index('resolution')
        word_rank = word_rank.loc[best_reso].reset_index().set_index('ngram word')
        best_words = []
        if( self.word_selection == 'p threshold' ):
            best_words = word_rank[word_rank['p'] <= self.p_threshold].index.values
        if( self.word_selection == 'best n words' ):
            best_words = word_rank.sort_values('rank', ascending=False).iloc[0:self.n_words].index.values

        self.selected_words = set(best_words)
        
        s = time.time()
        self.clf.fit(feature_vec[best_words], labels)
        spend = time.time() - s
        print('training time: {}'.format(spend))
        
        self._is_fitted = True
    
    def predict(self, data):
        
        self.check_is_fitted()
        
        # testing one specific dataset to avoid rework
        folder_path = 'C:/Users/marci/Desktop/MasterDegreeWorkspace/source/experiments/data_visualizing/worms/WordLen_6/'
        bob_path = folder_path+'/bag_of_bags_test.csv'
        bob = pd.read_csv(bob_path)
        
        
        # testing one specific resolution
        #matrix = self.resolution_matrix.matrix
        #window = matrix.columns[0]
        #ngram = matrix[matrix[window] > 0].index[0]
        #bob = bob[ bob.window == window]
        #bob = bob[ bob.ngram == ngram]
        
        #data = data.squeeze()                
        #bob = self._extract_bob_from(data)
        
        test_words = set(bob['ngram word'].unique())
        first_filtering = self.selected_words.intersection(test_words)
        print('intersecting words:', len(first_filtering))
        filtered_bob = bob.set_index('ngram word').loc[first_filtering].reset_index()
        
        all_samples = set(bob['sample'].unique())
        filtered_samples = set(filtered_bob['sample'].unique())
        missing_samples = all_samples - filtered_samples
        
        sample_group = bob.groupby('sample').head(1)
        for sample in missing_samples:
            filtered_bob = filtered_bob.append( sample_group[sample_group['sample'] == sample] )
        
        feature_vec = filtered_bob.pivot(index='sample', columns='ngram word', values='frequency')
        feature_vec = feature_vec.fillna(0)
        
        for word in self.selected_words:
            if word not in feature_vec.columns:
                feature_vec[word] = 0
        
        filtered_vec = feature_vec[self.selected_words]
        
        return self.clf.predict(filtered_vec)
        
    
    def _extract_bob_from(self, timeseries_set, labels=None):
        
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
        
        
        if labels is not None:
            bag_of_bags['label'] = labels.loc[bag_of_bags['sample']].values
        
        return bag_of_bags
