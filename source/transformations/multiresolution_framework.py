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
                 normalize = False,
                 remove_repeat_words=False,
                 discretization = "Multidomain"):

        self.resolution_matrix = resolution_matrix
        self.word_len = word_len
        self.alphabet_size = alphabet_size
        self.normalize = normalize 
        self.remove_repeat_words = remove_repeat_words
        self.discretization = discretization
        
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
                 disc = AdaptedSAX(word_length = self.word_len,
                            alphabet_size = self.alphabet_size,
                            window_size = window,
                            remove_repeat_words=self.remove_repeat_words,
                            return_pandas_data_series=True).fit(1)
                 self.sax_transformers.loc[window] = disc

            if self.sfa:
                disc = AdaptedSFA(word_length=self.word_len,
                           alphabet_size = self.alphabet_size,
                           window_size = window,
                           norm=self.normalize,
                           remove_repeat_words=self.remove_repeat_words,
                           return_pandas_data_series=True).fit(X, y)
                
                self.sfa_transformers.loc[window] = disc
        
        self._is_fitted = True
        return self        

    def transform(self, X, y=None):
                
        self.check_is_fitted()
        #TODO extend to multivariate
        idcs = X.index.values
        #X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)
        #X = X.squeeze(1)
        
        #TODO verify the time spend with DataFrame form
        word_sequences = pd.DataFrame()
        for window in self.windows:
            if self.sax:
                word_sequences = pd.concat(
                    [word_sequences, self.extract_word_sequences("SAX", window, X, idcs)],
                    ignore_index = True,
                    axis=0
                    )
            if self.sfa:
                word_sequences = pd.concat(
                    [word_sequences, self.extract_word_sequences("SFA", window, X, idcs)],
                    ignore_index = True,
                    axis=0
                    )
        
        return word_sequences
                
                
    def extract_word_sequences(self, discretization, window, X, indices, dim=None, y=None):
        
        discretizer = self.sax_transformers.loc[window]
        if(discretization=="SFA"):
            discretizer = self.sfa_transformers.loc[window]
            
        word_sequences = discretizer.transform(X, y)
        self._add_identifier(word_sequences, window, "window")
        self._add_identifier(word_sequences, discretization, "discretization")
        #TODO add the dimension identifier
        #self.add_identifier(word_sequences, dim, "dimension")
        word_sequences['sample'] = indices
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

    def _add_identifier(self, word_sequences, value, identifier):
        
        value_id = 0
        shift = 0
        if identifier == "window":
            value_id = self.fitted_windows.index(value)
            shift = self.window_bits
            
        elif identifier == "discretization":
            value_id = self.disc_id[value]
            shift = self.discretization_bits
            
        else:
            raise ValueError("The identifier {} is not expected in the "
                             "function _add_identifier".format(identifier))
            
        for dim in word_sequences:
            for sample in word_sequences[dim]:
                sample.apply( lambda w: ( w<<shift ) | value_id )
        
        
        
        
        
        
        
        
        
        
        