import sys
sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace/source')
sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace')

import pandas as pd
import numpy as np
from sktime.classification.base import BaseClassifier
from sklearn.linear_model import LogisticRegression

from source.utils import ResolutionMatrix
from sktime.transformations.panel.dictionary_based import SFA, SAX

from sklearn.feature_selection import chi2


class SearchTechnique_WS(BaseClassifier):
    """
        Window selection approach, based on the shotgun approach
    
    """
    
    def __init__(self,
                 word_length = 6,
                 alphabet_size = 4,
                 #word_ranking_method = 'chi2',
                 #word_selection = 'best n words', # ['p threshold', 'best n words']
                 #p_threshold = 0.05,
                 discretization = 'SFA',
                 max_num_windows = 20,
                 n_words = 200,
                 method = 'mean',
                 n_best_windows = 1,
                 normalize = True,
                 verbose = False,
                 random_state = None):
        
        
        #if (word_selection != 'p threshold') and (word_selection != 'best n words'):
        #    raise ValueError('The word selection must the a valid method of selection, as "p threshold" or "best n words"')
        
        self.word_length = word_length
        self.alphabet_size = alphabet_size
        self.max_num_windows = max_num_windows
        self.max_window_length = .5
        self.remove_repeat_words = False
        
        #self.p_threshold = p_threshold
        self.n_words = n_words
        self.method = method
        self.n_best_windows = n_best_windows
        self.normalize = normalize
        self.verbose = verbose
        
        
        self.random_state = random_state
        
        self.discretization = discretization
        self.discretizers = pd.Series()
        
        self.clf = LogisticRegression(max_iter=5000,
                                      random_state=random_state)
        
        self.ts_length = None
        self.windows = None
        self.results = pd.DataFrame()
        self.windows_index = []
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
        
        self.ts_length = data.iloc[0,0].size
        self.windows = ResolutionMatrix(self.ts_length,
                                        self.word_length,
                                        self.max_window_length,
                                        self.max_num_windows).matrix.columns.values
        
        if self.verbose:
            print('\nFitting the Classifier with data...')
        
            print('\nFitting the transformers...')
            for w in self.windows:
                print('_',end='')
            print('')
        for window in self.windows:
            if self.verbose:
                print('#', end='')
                
            if self.discretization == 'SFA':
                disc = SFA(window_size = window,
                           word_length=self.word_length,
                           alphabet_size=self.alphabet_size,
                           norm=self.normalize,
                           remove_repeat_words=self.remove_repeat_words,
                           return_pandas_data_series=True,
                           n_jobs=-1
                           ).fit(data, labels)
            else:
                disc = SAX(window_size = window,
                           word_length=self.word_length,
                           alphabet_size=self.alphabet_size,
                           remove_repeat_words=self.remove_repeat_words,
                           return_pandas_data_series=True
                           ).fit(data, labels)
                
            self.discretizers.loc[window] = disc
        
        bag_of_bags = self._extract_features(data, labels)
        bag_of_bags = self._window_selection(bag_of_bags, labels, self.n_words)
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
        for window in self.windows:
            if self.verbose:
                print('#', end='')
                
            disc = self.discretizers[window]
            word_sequence = disc.transform(data, labels)
            bag_of_words = self._get_feature_matrix(word_sequence)
            bag_of_words = self._add_identifier(bag_of_words, window)
            if labels is None:
                bag_of_words = self._feature_filtering(bag_of_words)
            else:
                bag_of_words = self._feature_selection(bag_of_words, labels, self.n_words)

            bob = pd.concat([bob, bag_of_words], axis=1)
            self.windows_index.append((window, bag_of_words.shape[1]))
        
        return bob

    def _add_identifier(self, bag_of_words, window):
        
        columns = bag_of_words.columns.map(lambda word: f'{window} {word}')
        bag_of_words.columns = columns
        return bag_of_words

    def _feature_selection(self, bag_of_words, labels, n_words):
        
        rank_value, p = chi2(bag_of_words, labels)
        word_rank = pd.DataFrame(index = bag_of_words.columns)
        word_rank['rank'] = rank_value
        word_rank = word_rank.sort_values('rank', ascending=False)
        best_words = word_rank.iloc[0:n_words].index.values        
        
        return bag_of_words[best_words]
    
    def _window_selection(self, bag_of_words, labels, n_words):
        
        rank_value, p = chi2(bag_of_words, labels)
        word_rank = pd.DataFrame(index = bag_of_words.columns)
        word_rank['rank'] = rank_value
        
        windows_index = []
        for window, qnt in self.windows_index:
            windows_index += [window]*qnt
        word_rank['window'] = windows_index

        best_windows = None
        if self.method == 'mean':
            best_windows = (
                word_rank
                .groupby('window')
                .mean()
                .sort_values('rank')
                .index[-self.n_best_windows:]
            )
        elif self.method == 'max':
            best_windows = (
                word_rank
                .groupby('window')
                .max()
                .sort_values('rank')
                .index[-self.n_best_windows:]
            )
        self.windows = best_windows
        best_words = word_rank[word_rank['window'].isin(best_windows)].index.values
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