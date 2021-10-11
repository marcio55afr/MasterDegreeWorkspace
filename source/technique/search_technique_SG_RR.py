import sys
sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace/source')
sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace')

import random
import pandas as pd
import numpy as np
from sktime.classification.base import BaseClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from source.utils import ResolutionHandler, ResolutionMatrix
from sktime.transformations.panel.dictionary_based import SFA, SAX
from sklearn.model_selection import cross_validate

from source.technique.word_ranking import WordRanking
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

from source.technique.resolution_selector import ResolutionSelector

from source.experiments.database import read_bob, write_bob


class SearchTechnique_SG_RR(BaseClassifier):
    """
        Shotgun approach with Random Resolution
    
    """
    
    def __init__(self,
                 #word_length = 6,
                 #alphabet_size = 4,
                 #word_ranking_method = 'chi2',
                 #word_selection = 'best n words', # ['p threshold', 'best n words']
                 #p_threshold = 0.05,
                 discretization = 'SFA',
                 num_resolutions = 20,
                 n_words = 200,
                 double_selection = False,
                 verbose = False,
                 random_state = None):
        
        
        #if (word_selection != 'p threshold') and (word_selection != 'best n words'):
        #    raise ValueError('The word selection must the a valid method of selection, as "p threshold" or "best n words"')
        
        self.num_resolutions = num_resolutions
        
        #self.p_threshold = p_threshold
        self.n_words = n_words
        self.double_selection = double_selection
        self.verbose = verbose
        
        
        self.random_state = random_state
        
        self.discretization = discretization
        self.discretizers = {}
        
        self.clf = LogisticRegression(max_iter=5000,
                                      random_state=random_state)
        
        self.alphabet_size = [2,3,4]
        self.word_lengths = [4,6,8]
        self.windows = None
        self.ts_length = None
        self.resolutions = None
        self.results = pd.DataFrame()
        self.selected_words = set()    
    
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
                
        if self.n_words<=0:
            raise ValueError('When select_features is selected as True '
                             'the n_words must be a positive number.')
            
        if data.iloc[0,0].size < self.word_lengths[-1]:
            raise ValueError(
                f'The length of the time series ({data.iloc[0,0].size}) must be '
                f'greater than the word_length ({self.word_lengths[-1]}'
            )
        
        self.ts_length = data.iloc[0,0].size
        random.seed(self.random_state)
        self.windows = np.arange(self.word_lengths[-1], self.ts_length//2)
        if self.windows.size == 0:
            self.windows = [self.word_lengths[-1]]
            
        resolutions = pd.Series([(a, l, w) for a in self.alphabet_size
                                 for l in self.word_lengths
                                 for w in self.windows])
        self.num_resolutions = min(len(resolutions), self.num_resolutions)
        self.resolutions = resolutions.sample(self.num_resolutions,
                                              random_state=self.random_state)
        if self.verbose:
            print('\nFitting the Classifier with data...')
        
            print('\nFitting the transformers...')
            for w in self.windows:
                print('_',end='')
            print('')
        for alphabet_size, word_len, window_size in self.resolutions:
            if self.verbose:
                print('#', end='')
                
            if self.discretization == 'SFA':
                disc = SFA(word_length = word_len,
                           window_size = window_size,
                           alphabet_size = alphabet_size,
                           norm = random.randint(0, 1),
                           remove_repeat_words = random.randint(0, 1),
                           return_pandas_data_series=True,
                           n_jobs=-1
                           ).fit(data, labels)
            else:
                disc = SAX(word_length = word_len,
                           window_size = window_size,
                           alphabet_size = alphabet_size,
                           remove_repeat_words = random.randint(0, 1),
                           return_pandas_data_series=True
                           ).fit(data, labels)
                
            self.discretizers[(alphabet_size, word_len, window_size)] = disc
        
        bag_of_bags = self._extract_features(data, labels)
        if self.double_selection:
            bag_of_bags = self._feature_selection(bag_of_bags, labels, self.n_words)
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
        for alphabet_size, word_len, window_size in self.resolutions:
            if self.verbose:
                print('#', end='')
                
            disc = self.discretizers[(alphabet_size, word_len, window_size)]
            word_sequence = disc.transform(data, labels)
            bag_of_words = self._get_feature_matrix(word_sequence)
            bag_of_words = self._add_identifier(bag_of_words, alphabet_size, word_len, window_size)
            if labels is None:
                bag_of_words = self._feature_filtering(bag_of_words)
            else:
                bag_of_words = self._feature_selection(bag_of_words, labels, self.n_words)
            bob = pd.concat([bob, bag_of_words], axis=1)
        
        return bob

    def _add_identifier(self, bag_of_words, alphabet_size, word_len, window):
        
        columns = bag_of_words.columns.map(lambda word: f'{alphabet_size} {word_len} {window} {word}')
        bag_of_words.columns = columns
        return bag_of_words

    def _feature_selection(self, bag_of_words, labels, n_words):
        
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
            bag_of_bags.insert(bag_of_bags.shape[1],missing_word,0)
        
        if self.verbose:
            print('Intersecting words: {}'.format( mask.sum()) )
        return bag_of_bags            
    
    def _get_feature_matrix(self, word_sequence):
        
        if self.discretization == 'SAX':
            word_counting = word_sequence[0].map( pd.value_counts )
            feature_matrix = pd.concat(list(word_counting), axis=1).T.fillna(0).astype(np.int16)
        else:
            feature_matrix = pd.concat(word_sequence[0].tolist(), axis=1).T.fillna(0).astype(np.int16)
        return feature_matrix

