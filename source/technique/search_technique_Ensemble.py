# -*- coding: utf-8 -*-

import sys
sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace/source')
sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace')

import os
import joblib
import pandas as pd
import numpy as np
from sktime.classification.base import BaseClassifier
from sklearn.ensemble import RandomForestClassifier

from source.transformations import AdaptedSAX, AdaptedSFA
#import multiprocessing as mp
#from sktime.transformations.panel.dictionary_based import SFA, SAX
#from multiprocessing import Pool
#import functools
from statistics import mode

from sklearn.feature_selection import chi2

class Selector(object):
    
    def __init__(self,
                 selector_id,
                 ts_length,
                 word_length,
                 alphabet_size,
                 method,
                 n_windows,
                 discretization='SFA',
                 random_state=None):
        
        self.selector_id = selector_id
        self.ts_length = ts_length
        self.word_length = word_length
        self.alphabet_size = alphabet_size
        self.method = method
        self.discretization = discretization
        
        self.N =  3
        self.n_words = 200
        self.remove_repeat_words = False
        self.normalize = True
        self.random_state = random_state
        
        self.rng = np.random.default_rng(random_state)
        self.discretizers = pd.Series()
        self.n_windows = min(ts_length//2 - word_length + 1, n_windows)
        self.windows = self._get_windows()
        self.ngrams = self._get_ngrams()
        self.clf =  RandomForestClassifier(criterion="gini",
                                           n_estimators = 1000,
                                           class_weight='balanced_subsample',
                                           n_jobs=-1,
                                           random_state=random_state)
        self._is_fitted = False
        
    def fit(self, data, labels):
        
        self._fit_discs(data, labels)     
        bag_of_bags = self._extract_features(data, labels)            
        self.selected_words = bag_of_bags.columns.values
        self.clf.fit(bag_of_bags, labels)
        self._is_fitted = True
        #self._save_model()
        
    def predict(self, data):
                
        #self._load_model()
        self._check_is_fitted()
        bag_of_bags =  self._extract_features(data, None)            
        bag_of_bags = self._feature_fixing(bag_of_bags)
        predictions = self.clf.predict(bag_of_bags)
        return predictions
    
    def predict_proba(self, data):
        
        #self._load_model()
        self._check_is_fitted()
        bag_of_bags =  self._extract_features(data, None)            
        bag_of_bags = self._feature_fixing(bag_of_bags)
        probabilities = self.clf.predict_proba(bag_of_bags)
        return probabilities
    
    def _get_windows(self):
        
        min_window_size = self.word_length
        max_window_size = self.ts_length//2
        windows = self.rng.choice(max_window_size-min_window_size+1,
                                  self.n_windows, replace=False)
        
        return windows + min_window_size
    
    def _get_ngrams(self):    
        return self.rng.choice(3, self.n_windows) + 1
        
    def _fit_discs(self, data, labels ):
        
        if self.discretization == 'SFA':
            for window in self.windows:
                sfa = AdaptedSFA(window_size = window,
                                 word_length=self.word_length,
                                 alphabet_size=self.alphabet_size,
                                 norm=self.normalize,
                                 remove_repeat_words=self.remove_repeat_words,
                                 return_pandas_data_series=False,
                                 n_jobs=-1
                                 ).fit(data, labels)
                self.discretizers.loc[window] = sfa
        else:
            for window in self.windows:
                sax = AdaptedSAX(window_size = window,
                                 word_length=self.word_length,
                                 alphabet_size=self.alphabet_size,
                                 remove_repeat_words=self.remove_repeat_words,
                                 return_pandas_data_series=False
                                 ).fit(data, labels)                
                self.discretizers.loc[window] = sax
    
    def _extract_features(self, data, labels):
        
        bob = pd.DataFrame()
        if self.discretization == 'SFA':
            for window in self.windows:
                
                for n in range(self.N):
                    disc = self.discretizers[window]
                    word_sequence = disc.transform(data, labels)
                    ngram_sequence = self._extract_ngram_words(n, word_sequence)
                    bag_of_words = self._get_feature_matrix(ngram_sequence)
                    bag_of_words = self._add_identifier(bag_of_words, window)
                    if labels is None:
                        bag_of_words = self._feature_filtering(bag_of_words)
                    else:
                        bag_of_words = self._feature_selection(bag_of_words, labels, self.n_words)
                    bob = pd.concat([bob, bag_of_words], axis=1)
        else:
            for window in self.windows:
                
                for n in range(self.N):
                    disc = self.discretizers[window]
                    word_sequence = disc.transform(data, labels)
                    ngram_sequence = self._extract_ngram_words(n, word_sequence)
                    bag_of_words = self._get_feature_matrix(ngram_sequence)
                    bag_of_words = self._add_identifier(bag_of_words, window)
                    if labels is None:
                        bag_of_words = self._feature_filtering(bag_of_words)
                    else:
                        bag_of_words = self._feature_selection(bag_of_words, labels, self.n_words)
                    bob = pd.concat([bob, bag_of_words], axis=1)
        
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
            
        return bag_of_bags[self.selected_words]      
    
    def _get_feature_matrix(self, ngram_sequences):
        
        ngram_counts = list(map(pd.value_counts, ngram_sequences))
        bag_of_words = pd.concat(ngram_counts, axis=1).T  
        return bag_of_words.fillna(0).astype(np.int32)    
        
    def _extract_ngram_words(self, n, word_sequences):
        
        dim_0 = 0
        word_sequences = word_sequences[dim_0]
        word_sequences = list( map(lambda sequence: list(map(str,sequence)), word_sequences) )
        if n == 0:
            return word_sequences
        
        n += 1
        ngram_sequence = []
        for sequence in word_sequences:
            ngrams = []
            for i in range(n):
                ngrams += zip(*[iter(sequence[i:])]*n)
            ngram_sequence.append( list(map(' '.join, ngrams)) )
            
        return ngram_sequence
     
    def _check_is_fitted(self):
        
        if self._is_fitted == False:
            raise RuntimeError('The Selector is not fitted yet to do that')
            
    def _save_model(self):       
        if not os.path.exists('temp'):
            os.mkdir('temp')
            os.mkdir('temp/selectors')            
        path = f'temp/selectors/selector_{self.selector_id}/'
        self._save_clf(path)
        self._save_discs(path)
    
    def _load_model(self):            
        path = f'temp/selectors/selector_{self.selector_id}/'
        if (not os.path.exists('temp')) or (not os.path.exists('temp/selectors')):
            raise RuntimeError('The Selector is not fitted')        
        self._load_clf(path)
        self._load_discs(path)       
    
    def _save_clf(self,path):
        file = path + 'rf.joblib'
        os.mkdir(path)
        joblib.dump(self.clf, file, compress=0) 
    
    def _load_clf(self,path):
        file = path + 'rf.joblib'
        self.clf = joblib.load(file)
        self._is_fitted = True
    
    def _save_discs(self,path):
        for window in self.discretizers.index:
            file = path+ f'disc_{window}.joblib'
            joblib.dump(self.discretizers.loc[window], file, compress=0) 
    
    def _load_discs(self,path):
        for window in self.discretizers.index:
            file = path+ f'disc_{window}.joblib'
            if not os.path.isfile(file):
                raise RuntimeError(f'There is no discretizer with window {window}')
            self.discretizers.loc[window] = joblib.load(file)
        
        
        
        
class SearchTechnique_Ensemble(BaseClassifier):
    """
        Ensemble approach of the NgramResolution
    
    """
    
    def __init__(self,
                 num_clfs,
                 N = 3,
                 word_length = 4,
                 alphabet_size = 4,
                 max_window_length = .5,
                 sfa_window_per_slc = 8,
                 sax_window_per_slc  = 2,
                 n_sfa_words = 200,
                 n_sax_words = 200,
                 method=1,
                 normalize = True,
                 verbose = False,
                 random_state = None):
        
        
        #if (word_selection != 'p threshold') and (word_selection != 'best n words'):
        #    raise ValueError('The word selection must the a valid method of selection, as "p threshold" or "best n words"')
        
        self.num_clfs = num_clfs
        self.n_sax_slc = num_clfs//3
        self.n_sfa_slc = num_clfs - self.n_sax_slc
        self.data_frac = 2*1/self.num_clfs
        
        self.N = N
        self.word_length = word_length
        self.alphabet_size = alphabet_size
        self.max_window_length = max_window_length
        
        self.sfa_window_per_slc = sfa_window_per_slc
        self.sax_window_per_slc = sax_window_per_slc
        self.n_sfa_words = n_sfa_words
        self.n_sax_words = n_sax_words
        self.method = method
        
        self.normalize = normalize
        self.verbose = verbose
        self.random_state = random_state
        #self.n_jobs = n_jobs

        self.sfa_discretizers = pd.Series()
        self.sax_discretizers = pd.Series()        
        
        self.rng = np.random.default_rng(random_state)
        self.remove_repeat_words = False
        self.ts_length = None
        self.windows = None
        self.results = pd.DataFrame()
        self.selected_words = set()
        self.sfa_id = 0
        self.sax_id = 1
        
        self.ensemble = []
        self.ensemble_is_created = False
        self.slc = []

    
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
        
        if self.method == 2:
            self.sfa_window_per_slc = 1
            self.sax_window_per_slc = 1
            self.n_sax_slc = self.num_clfs//5
            self.n_sfa_slc = self.num_clfs - self.n_sax_slc
            self.data_frac = 0.05
        
        
        ts_length = data.iloc[0,0].size        
        self._create_ensemble(ts_length)
        self._fit_ensemble(data, labels)        
        self._is_fitted = True
    
    def predict(self, data):
        
        if self.verbose:
            print('Predicting data with the Classifier...\n')
        
        self.check_is_fitted()
        results = [selector.predict(data) for selector in self.ensemble]        
        predictions = [mode(e) for e in zip(*results)]
        return predictions
        
    
    def predict_proba(self, data):
        
        if self.verbose:
            print('Predicting data with the Classifier...\n')
        
        self.check_is_fitted()        
        results = [selector.predict_proba(data) for selector in self.ensemble]        
        probabilities = np.mean( np.array(results), axis=0 )        
        return probabilities
    
    def _create_ensemble(self, ts_length):
        
        rand_ints = self.rng.integers(low=0, high=1000, size=self.n_sfa_slc)
        sfa_selectors = [Selector(selector_id = i,
                                  n_windows = self.sfa_window_per_slc,
                                  ts_length = ts_length,
                                   word_length = self.word_length,
                                   alphabet_size = self.alphabet_size,
                                   discretization='SFA',
                                   method = 1,
                                   random_state = rand_ints[i]) 
                         for i in range(self.n_sfa_slc)]
        
        rand_ints = self.rng.integers(low=0, high=1000, size=self.n_sax_slc)
        sax_selectors = [Selector(selector_id = i+self.n_sfa_slc,
                                  n_windows = self.sax_window_per_slc,
                                  ts_length = ts_length,
                                  word_length = self.word_length,
                                  alphabet_size = self.alphabet_size,
                                  discretization='SAX',
                                  method = 1,
                                  random_state = rand_ints[i]) 
                         for i in range(self.n_sax_slc)]
        
        self.ensemble = sfa_selectors + sax_selectors
        self.ensemble_is_created = True
        

    def _fit_ensemble(self, data, labels):
        
        if not self.ensemble_is_created:
            raise RuntimeError('Ensemble not create to be fitted')        
        
        data['target'] = labels
        rand_ints = self.rng.integers(low=0, high=1000, size=self.num_clfs)
        
        samples = []
        classes = np.unique(labels)
        for class_ in classes:
            data_class = data[data['target'] == class_]
            size = np.int16(np.ceil(data_class.shape[0]*self.data_frac))
            samples_class = [ data_class.sample(size,
                                   random_state=random_state) for random_state in rand_ints]
            samples.append( samples_class)
        samples = zip(*samples)
        samples = list(map(pd.concat, samples))
        
        #pool = []  
        for i in range(self.num_clfs):
            sample = samples[i]
            y = sample['target']
            X = sample.drop('target', axis=1)
            selector = self.ensemble[i]
            selector.fit(X,y)
            #pool += [mp.Process(target=selector.fit, args=(X,y))]
        '''
        for p in pool[:2]:
            p.start()
        for p in pool[:2]:
            p.join()
        input('Helllo')
        
        pool.close'''



















        
        
        
        