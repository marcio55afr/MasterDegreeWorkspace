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
from source.transformations import MultiresolutionFramework

from source.technique.word_ranking import WordRanking
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

from source.technique.resolution_selector import ResolutionSelector

from source.experiments.database import read_bob, write_bob
from source.experiments.database.bob_handler import read_bag, write_bag


class SearchTechnique(BaseClassifier):
    
    
    
    def __init__(self,
                 word_length = 6,
                 alphabet_size = 4,
                 word_ranking_method = 'chi2',
                 word_selection = 'best n words', # ['p threshold', 'best n words']
                 p_threshold = 0.05,
                 n_words = 120,
                 num_windows = None,
                 max_window_length = None,
                 normalize = True,
                 verbose = True,
                 random_state = None):
        
        
        if (word_selection != 'p threshold') and (word_selection != 'best n words'):
            raise ValueError('The word selection must the a valid method of selection, as "p threshold" or "best n words"')
        
        self.word_length = word_length
        self.alphabet_size = alphabet_size
        self.num_windows = num_windows
        self.max_window_length = max_window_length
        
        self.word_ranking_method = word_ranking_method
        self.word_selection = word_selection
        self.p_threshold = p_threshold
        self.n_words = n_words
        self.normalize = normalize
        self.verbose = verbose
        
        
        self.discretization = "SFA"        
        self.random_state = random_state
        self.clf = LogisticRegression(max_iter=5000,
                                      random_state=self.random_state)
        
        self.ts_length = None
        self.resolution_matrix = None
        self.framework = None        
        self.selected_words = set()
    
        # test variables
        # self.resolution_matrix_aux = None
    
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
        
        if self.verbose:
            print('Creating the Resolution Matrix...\n')
        
        ts_length = data.iloc[0,0].size
        self.ts_length = ts_length
        # TODO implement the function create_matrix in the ResolutionHandler
        # no need for this ResolutionMatrix class
        self.resolution_matrix = ResolutionMatrix(self.ts_length,
                                                  self.word_length,
                                                  self.num_windows,
                                                  self.max_window_length)
        
        self.framework = MultiresolutionFramework(self.resolution_matrix.matrix,
                                                  word_len=self.word_length,
                                                  alphabet_size=self.alphabet_size,
                                                  discretization=self.discretization,
                                                  normalize=self.normalize,
                                                  verbose=self.verbose)
                
        if self.verbose:
            print('Fitting the Classifier with data...\n')
        
        # testing one specific dataset to avoid rework
        #try:
        #    bob, samples_id, words = read_bob( 'ecg', self.discretization, self.word_length, 'train' )
        #    #bag = read_bag( 'ecg', self.discretization, self.word_length, 'train' )
        #    self.framework.fit(data, labels)
        #except Exception as e:
        #    print(e)
        #    bob, samples_id, words = self._extract_bob_from(data, labels)
        #    write_bob( bob, samples_id, words, 'ecg', self.discretization, self.word_length, 'train' )
            #bag = self._extract_bob_from(data, labels)
            #write_bag(bag, 'ecg', self.discretization, self.word_length, 'train' )
        
        # testing one specific resolution
        #bob = bag
        #matrix = self.resolution_matrix_aux
        #window = matrix.columns[0]
        #ngram = matrix[matrix[window] > 0].index[0]
        #bob = bob[ bob.window == window]
        #bob = bob[ bob.ngram == ngram]
        #bob.index.name = 'word'
        #bob = bob.pivot_table(values='frequency',
        #                      columns='word',
        #                      index='sample').fillna(0).astype(np.int32)
        #words = bob.columns.values
        #samples_id = bob.index.values
                                
        bob, samples_id, words = self._extract_bob_from(data, labels)
        
        
        rank_value, p = chi2(bob, labels)
        word_rank = pd.DataFrame(index = words)
        word_rank['rank'] = rank_value
        word_rank['p'] = p
        word_rank.index.name = 'word'
        
        #resolution = word_rank.index.map( lambda x: ResolutionHandler.get_resolution_from(x) )
        #word_rank['resolution'] = resolution        
        #best_reso = ResolutionSelector.get_best_resolution_max(word_rank)
        
        #word_rank = word_rank.reset_index().set_index('resolution')
        #word_rank = word_rank.loc[best_reso].reset_index().set_index('ngram word')
        best_words = []
        if( self.word_selection == 'p threshold' ):
            best_words = word_rank[word_rank['p'] <= self.p_threshold].index.values
            print('words above the p threshold: ', len(best_words))
        if( self.word_selection == 'best n words' ):
            best_words = word_rank.sort_values('rank', ascending=False).iloc[0:self.n_words].index.values

        self.selected_words = sorted(best_words)
        
        feature_vec = pd.DataFrame(index=samples_id)
        indices = word_rank.index.get_indexer(self.selected_words)
        for i,j in zip(range(indices.size),indices):
            feature_vec[self.selected_words[i]] = bob.getcol(j).toarray().squeeze()
            #feature_vec[self.selected_words[i]] = bob[self.selected_words[i]]
            
        s = time.time()
        self.clf.fit(feature_vec, labels)
        spend = time.time() - s
        print('LR training time: {}'.format(spend))
        
        self._is_fitted = True
    
    def predict(self, data):
        
        if self.verbose:
            print('Predicting data with the Classifier...\n')
        self.check_is_fitted()
        
        # testing one specific dataset to avoid rework
        #try:
        #    bob, samples_id, words = read_bob( 'ecg', self.discretization, self.word_length, 'test' )
            #bag = read_bag( 'ecg', self.discretization, self.word_length, 'test' )
        #except:
        #    bob, samples_id, words  = self._extract_bob_from(data)
        #    write_bob( bob, samples_id, words, 'ecg', self.discretization, self.word_length, 'test' )
            #bag = self._extract_bob_from(data)
            #write_bag( bag, 'ecg', self.discretization, self.word_length, 'test' )
            
        
        # testing one specific resolution
        #bob = bag
        #matrix = self.resolution_matrix.matrix
        #window = matrix.columns[0]
        #ngram = matrix[matrix[window] > 0].index[0]
        #bob = bob[ bob.window == window]
        #bob = bob[ bob.ngram == ngram]
        #bob.index.name = 'word'
        #bob = bob.pivot_table(values='frequency',
        #                      columns='word',
        #                      index='sample').fillna(0).astype(np.int32)
        #words = bob.columns.values        
        #samples_id = bob.index.values
        
        bob, samples_id, words = self._extract_bob_from(data)
        
        intersecting_words = []
        
        words_index = pd.Index(words)
        indices = words_index.get_indexer(self.selected_words)
        feature_vec = pd.DataFrame(index=samples_id)
        for i,j in zip(range(indices.size),indices):
            if j>=0:
                feature_vec[self.selected_words[i]] = bob.getcol(j).toarray().squeeze()
                #feature_vec[self.selected_words[i]] = bob[self.selected_words[i]]
                intersecting_words.append(self.selected_words[i])
            else:
                feature_vec[self.selected_words[i]] = 0
        
        print('Intersecting words: {}'.format( len(intersecting_words)) )
        
        #all_samples = set(bob['sample'].unique())
        #filtered_samples = set(filtered_bob['sample'].unique())
        #missing_samples = all_samples - filtered_samples
        
        #sample_group = bob.groupby('sample').head(1)
        #for sample in missing_samples:
        #    filtered_bob = filtered_bob.append( sample_group[sample_group['sample'] == sample] )
        
        #feature_vec = filtered_bob.pivot(index='sample', columns='ngram word', values='frequency')
        #feature_vec = feature_vec.fillna(0)
        
        
        #filtered_vec = feature_vec[self.selected_words]
        
        return self.clf.predict(feature_vec)
    
    def predict_proba(self, data):
        
        if self.verbose:
            print('Predicting data with the Classifier...\n')
        self.check_is_fitted()
        
        bob, samples_id, words = self._extract_bob_from(data)
        
        intersecting_words = []
        
        words_index = pd.Index(words)
        indices = words_index.get_indexer(self.selected_words)
        feature_vec = pd.DataFrame(index=samples_id)
        for i,j in zip(range(indices.size),indices):
            if j>=0:
                feature_vec[self.selected_words[i]] = bob.getcol(j).toarray().squeeze()
                intersecting_words.append(self.selected_words[i])
            else:
                feature_vec[self.selected_words[i]] = 0
        
        print('Intersecting words: {}'.format( len(intersecting_words)) )
    
    def _extract_bob_from(self, data, labels=None):        
        
        word_sequences = self.framework.transform(data) if labels is None \
            else self.framework.fit_transform(data, labels)
        bag_of_bags, samples_id, words = NgramExtractor.get_bob(word_sequences, self.resolution_matrix.matrix)
        return bag_of_bags, samples_id, words
        #bag_of_bags = NgramExtractor.get_bob(word_sequences, self.resolution_matrix.matrix)
        #return bag_of_bags
