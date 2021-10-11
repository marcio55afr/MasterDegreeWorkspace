# -*- coding: utf-8 -*-

import sys

import pandas as pd
import numpy as np

from sktime.transformations.base import _PanelToPanelTransformer
from source.transformations import AdaptedSAX, AdaptedSFA

#    TODO: verify this returned pandas is consistent with sktime
#    definition. Timestamps?
from sktime.utils.validation.panel import check_X

class MultiresolutionFramework(_PanelToPanelTransformer):

    _tags = {"univariate-only": True}

    def __init__(self,
                 resolution_matrix,
                 word_len = 6,
                 alphabet_size = 4,
                 remove_repeat_words=False,
                 discretization = "Multidomain",
                 normalize = True,
                 verbose=True):

        self.resolution_matrix = resolution_matrix
        self.word_len = word_len
        self.alphabet_size = alphabet_size
        self.remove_repeat_words = remove_repeat_words
        self.discretization = discretization
        self.normalize = normalize 
        self.verbose = verbose
        
        if self.verbose:
            print('Initializating the framework and its discretizers...\n')
        
        mask = self.resolution_matrix.sum() > 0
        self.windows = self.resolution_matrix.columns[mask].values
        self.fitted_windows = list(self.windows)
        
        self.valid_discretizations = ["SAX","SFA","Multidomain"]
        self.sax = True
        self.sfa = True
        if self.discretization == "SAX":
            self.sfa = False
        if self.discretization == "SFA":
            self.sax = False
        self.disc_id = {"SAX":0,
                        "SFA":1}
        
        self.sax_transformers = pd.Series()
        self.sfa_transformers = pd.Series()
        
        self.window_bits = np.int16(np.ceil(np.log2(len(self.fitted_windows))))
        self.discretization_bits = np.int16(1)

        super(MultiresolutionFramework, self).__init__()

    def fit(self, X, y=None):
        
        if y is None:
            if self.sfa == True:
                raise ValueError("The method using the sfa discretization must"
                                 " receive the labels of the samples")
        else:
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    "The number of labels must be equal to the number of samples."
                    "It was receive {} samples and {} labels".format(X.shape[0],
                                                                     y.shape[0])
                )

            if (X.index.values != y.index.values).any():
                raise ValueError("The indices of the samples in the X must be"
                                 " the same indices of the labels")
        
        if self.resolution_matrix.shape[0] > 5:
            raise ValueError(
                "The maximmum ngram supported is 5. "
                "This already creates too many possible words."
            )
            
        if not self.resolution_matrix.columns.is_monotonic_increasing:
            raise ValueError(
                "The window represented by the resolution matrix columns "
                "must be ordered and increasing"
                )
        
        if self.resolution_matrix.columns[0] < self.word_len:
            raise ValueError(
                "The smallest window must be bigger than the word length and "
                "it was received smallest window = "
                "{} and word length = {}".format(self.resolution_matrix.columns[0],
                                                 self.word_len)
                )
            
        if self.alphabet_size < 2 or self.alphabet_size > 4:
            raise ValueError("Alphabet size must be an integer between 2 and 4")
        
        if self.discretization not in self.valid_discretizations:
            raise ValueError(
                "All discretizations must be one of the: ",
                self.valid_discretizations
            )
                
        if self.windows.size == 0:
            raise ValueError(
                "The resolution matrix must have at least one valid resolution"
                " represented by a positive interger number"
            )
        
        #TODO extend to multivariate
        #X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)
        #X = X.squeeze(1)  
        
        for window in self.windows:
            if self.sax:
                 disc = AdaptedSAX(window_size = window,
                            word_length = self.word_len,
                            alphabet_size = self.alphabet_size,
                            remove_repeat_words=self.remove_repeat_words,
                            return_pandas_data_series=True).fit(1)
                 self.sax_transformers.loc[window] = disc

            if self.sfa:
                disc = AdaptedSFA(window_size = window,
                           word_length=self.word_len,
                           alphabet_size = self.alphabet_size,
                           norm=self.normalize,
                           remove_repeat_words=self.remove_repeat_words,
                           return_pandas_data_series=True).fit(X, y)
                
                self.sfa_transformers.loc[window] = disc
        
        self._is_fitted = True
        return self        

    def transform(self, X, y=None):
        
        if y is not None:
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    "The number of labels must be equal to the number of samples."
                    "It was receive {} samples and {} labels".format(X.shape[0],
                                                                     y.shape[0])
                )

            if (X.index.values != y.index.values).any():
                raise ValueError("The indices of the samples in the X must be"
                                 " the same indices of the labels")
                
        self.check_is_fitted()
        #TODO extend to multivariate        
        #TODO verify the time spend with DataFrame form
        word_sequences = pd.DataFrame()
        if self.verbose:
            print('Discretizing the time series...')
            for v in range(len(self.windows)):
                print('_', end='')
            print('')
        for window in self.windows:
            if self.sax:
                word_sequences = pd.concat(
                    [word_sequences, self.extract_word_sequences("SAX", window, X)],
                    ignore_index = False,
                    axis=0
                    )
            if self.sfa:
                word_sequences = pd.concat(
                    [word_sequences, self.extract_word_sequences("SFA", window, X)],
                    ignore_index = False,
                    axis=0
                    )
            if self.verbose:
                print('#', end='')
        if self.verbose:
            print('')
        
        return word_sequences
                
                
    def extract_word_sequences(self, discretization, window, X, dim=None, y=None):
        
        discretizers = self.sax_transformers if discretization=="SAX" else self.sfa_transformers
        discretizer = discretizers.loc[window]
            
        word_sequences = discretizer.transform(X, y)
        self._add_identifier(word_sequences, discretization, window)
        #self._add_identifier(word_sequences, discretization, "discretization")
        #TODO add the dimension identifier
        #self.add_identifier(word_sequences, dim, "dimension")
        word_sequences['window'] = window
        return word_sequences
        
        
    def update_resolution_matrix(self, resolution_matrix):
        self.resolution_matrix = resolution_matrix
        mask = self.resolution_matrix.sum() > 0
        self.windows = self.resolution_matrix.columns[mask].values
        
        if not self.resolution_matrix.columns.is_monotonic_increasing:
            raise ValueError(
                "The window represented by the resolution matrix columns "
                "must be ordered and increasing")

        if self.windows.size == 0:
            raise ValueError(
                "The resolution matrix must have at least one valid resolution"
                " represented by a positive interger number"
            )
        invalid_windows = [ w for w in self.windows if w not in self.fitted_windows]
        if invalid_windows:
            raise ValueError(
                "The windows of the new resolution matrix must be the same "
                "of the fitted windows"
                )

    def _add_identifier(self, word_sequences, discretization, window):
        
        # TODO extends it to a multivariate process
        dim = 0
        disc_id = self.disc_id[discretization]
        n = word_sequences.shape[0]
        for i in range(n):
            sample = word_sequences.loc[i,dim]
            word_sequences.loc[i,dim] = sample.apply( lambda w: '{} {} {}'.format(disc_id,
                                                                                  window,
                                                                                  w)
                                                     ).values
        
        
        
        
        
        
        
        
        
        
        