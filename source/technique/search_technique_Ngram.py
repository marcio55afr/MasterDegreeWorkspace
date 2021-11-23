import sys
sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace/source')
sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace')

import pandas as pd
import numpy as np
from sktime.classification.base import BaseClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from source.utils import ResolutionHandler, ResolutionMatrix
from source.transformations import AdaptedSAX, AdaptedSFA

from sklearn.feature_selection import chi2




class SearchTechnique_Ngram(BaseClassifier):
    """
        Multiresolution approach with the Shotgun using Random Forest as clf
    
    """
    
    def __init__(self,
                 N = 5,
                 word_length = 6,
                 alphabet_size = 4,
                 max_window_length = .5,
                 max_sfa_windows = 8,
                 max_sax_windows = 2,
                 n_sfa_words = 200,
                 n_sax_words = 200,
                 random_selection = True,
                 normalize = True,
                 verbose = False,
                 random_state = None):
        
        
        #if (word_selection != 'p threshold') and (word_selection != 'best n words'):
        #    raise ValueError('The word selection must the a valid method of selection, as "p threshold" or "best n words"')
        
        self.N = N
        self.word_length = word_length
        self.alphabet_size = alphabet_size
        self.max_window_length = max_window_length

        self.max_sfa_windows = max_sfa_windows
        self.max_sax_windows = max_sax_windows
        self.n_sfa_words = n_sfa_words
        self.n_sax_words = n_sax_words
        
        self.random_selection = random_selection
        self.normalize = normalize
        self.verbose = verbose        
        self.random_state = random_state
        
        self.sfa_discretizers = pd.Series()
        self.sax_discretizers = pd.Series()        
        
        self.clf =  RandomForestClassifier(criterion="gini",
                                               n_estimators = 1000,
                                               class_weight='balanced_subsample',
                                               n_jobs=-1,
                                               random_state=random_state)
        
        # Internal Variables
        self.remove_repeat_words = False
        self.ts_length = None
        self.windows = None
        self.results = pd.DataFrame()
        self.selected_words = set()    
        self.sfa_id = 0
        self.sax_id = 1
    
    def fit(self, data, labels):
        
        if type(data) != pd.DataFrame:
            raise TypeError("The data must be a type of pd.DataFrame."
                            " It was received as {}".format(type(data)))
                            
        if type(labels) != pd.Series:
            raise TypeError("The data must be a type of pd.Series."
                            " It was received as {}".format(type(labels)))
                            
        if data.shape[0] != labels.shape[0]:
            raise RuntimeError('The labels isn\'t compatible with the data received')
        
        if labels is not None:
            if (data.index.values != labels.index.values).any():
                raise RuntimeError("The samples indices must be the equal to "
                                   "the labels indices")
        
        self.ts_length = data.iloc[0,0].size
        self.sfa_windows = ResolutionMatrix(self.ts_length,
                                            self.word_length,
                                            self.max_window_length,
                                            self.max_sfa_windows).matrix.columns.values

        self.sax_windows = ResolutionMatrix(self.ts_length,
                                            self.word_length,
                                            self.max_window_length,
                                            self.max_sax_windows).matrix.columns.values
        
        if self.verbose:
            print('\nFitting the Classifier with data...')
        
            print('\nFitting the transformers...')
            for w in self.sfa_windows:
                print('_',end='')
            for w in self.sax_windows:
                print('_',end='')
            print('')
        for window in self.sfa_windows:
            if self.verbose:
                print('#', end='')
            sfa = AdaptedSFA(window_size = window,
                       word_length=self.word_length,
                       alphabet_size=self.alphabet_size,
                       norm=self.normalize,
                       remove_repeat_words=self.remove_repeat_words,
                       return_pandas_data_series=False,
                       n_jobs=-1
                       ).fit(data, labels)
            self.sfa_discretizers.loc[window] = sfa
        
        for window in self.sax_windows:
            if self.verbose:
                print('#', end='')
            sax = AdaptedSAX(window_size = window,
                       word_length=self.word_length,
                       alphabet_size=self.alphabet_size,
                       remove_repeat_words=self.remove_repeat_words,
                       return_pandas_data_series=False
                       ).fit(data, labels)                
            self.sax_discretizers.loc[window] = sax        
        
        bag_of_bags = self._extract_features(data, labels)
        self.selected_words = bag_of_bags.columns.values
        self.clf.fit(bag_of_bags, labels)
        self._is_fitted = True
    
    def predict(self, data):
        
        if self.verbose:
            print('Predicting data with the Classifier...\n')
        
        self.check_is_fitted()
        
        bag_of_bags = self._extract_features(data, None)
        bag_of_bags = self._feature_fixing(bag_of_bags)
        return self.clf.predict(bag_of_bags)
    
    def predict_proba(self, data):
        
        if self.verbose:
            print('Predicting data with the Classifier...\n')
        
        self.check_is_fitted()
                
        bag_of_bags = self._extract_features(data, None)
        bag_of_bags = self._feature_fixing(bag_of_bags)
        return self.clf.predict_proba(bag_of_bags)
    
    def _extract_features(self, data, labels):
        if self.verbose:
            print('\nExtracting features from all resolutions...')
            for w in self.windows:
                print('_',end='')
            print('')            
        
        bob = pd.DataFrame()
        
        for window in self.sfa_windows:
            if self.verbose:
                print('#', end='')
            
            disc = self.sfa_discretizers[window]
            word_sequence = disc.transform(data, labels)
            ngram_sequence = self._extract_ngram_words(word_sequence)
            bag_of_words = self._get_feature_matrix(ngram_sequence)
            bag_of_words = self._add_identifier(bag_of_words, self.sfa_id, window)
            if labels is None:
                bag_of_words = self._feature_filtering(bag_of_words)
            else:
                bag_of_words = self._feature_selection(bag_of_words, labels, self.n_sfa_words)

            bob = pd.concat([bob, bag_of_words], axis=1)
        
        for window in self.sax_windows:
            if self.verbose:
                print('#', end='')
            
            disc = self.sax_discretizers[window]
            word_sequence = disc.transform(data, labels)
            ngram_sequence = self._extract_ngram_words(word_sequence)
            bag_of_words = self._get_feature_matrix(ngram_sequence)
            bag_of_words = self._add_identifier(bag_of_words, self.sax_id, window)
            if labels is None:
                bag_of_words = self._feature_filtering(bag_of_words)
            else:
                bag_of_words = self._feature_selection(bag_of_words, labels, self.n_sax_words)

            bob = pd.concat([bob, bag_of_words], axis=1)
        
        return bob
    
    def _add_identifier(self, bag_of_words, disc_id, window):
        
        columns = bag_of_words.columns.map(lambda word: f'{disc_id} {window} {word}')
        bag_of_words.columns = columns
        return bag_of_words

    def _feature_selection(self, bag_of_words, labels, n_words):
        
        if self.random_selection:
            bag_of_words = bag_of_words.sample(frac=.5, axis=1)
        rank_value, p = chi2(bag_of_words, labels)
        word_rank = pd.DataFrame(index = bag_of_words.columns)
        word_rank['rank'] = rank_value
        word_rank = word_rank.sort_values('rank', ascending=False)
        best_words = word_rank.iloc[0:n_words].index.values        
        
        return bag_of_words[best_words]    

    def _feature_filtering(self, bag_of_words):
        
        indices = bag_of_words.columns.get_indexer(self.selected_words)
        mask = indices >= 0
        intersecting_words = self.selected_words[mask]
        bag_of_words = bag_of_words[ intersecting_words ]
        return bag_of_words
    
    def _feature_fixing(self, bag_of_bags):
        
        indices = bag_of_bags.columns.get_indexer(self.selected_words)
        mask = indices >= 0
        for missing_word in self.selected_words[~mask]:
            bag_of_bags[missing_word] = 0
        
        if self.verbose:
            print('Intersecting words: {}'.format( mask.sum()) )
        return bag_of_bags[self.selected_words]   
    
    def _get_feature_matrix(self, ngram_sequences):
                
        ngram_counts = list(map(pd.value_counts, ngram_sequences))
        bag_of_words = pd.concat(ngram_counts, axis=1).T  
        return bag_of_words.fillna(0).astype(np.int32)
    
    def _extract_ngram_words(self, word_sequences):
        
        def get_ngrams_from(sample_sequence):
            sample_sequence = list(map(str, sample_sequence))
            ngrams = []
            for n in range(1,self.N):
                n += 1
                for i in range(n):
                    ngrams += zip(*[iter(sample_sequence[i:])]*n)
            
            return sample_sequence + list(map(' '.join, ngrams))
        
        dim_0 = 0
        word_sequences = word_sequences[dim_0]        
        ngram_sequences = map(get_ngrams_from, word_sequences)
        return ngram_sequences
    
        





