# -*- coding: utf-8 -*-
import sys

import numpy as np
import pandas as pd
import scipy.stats

from sktime.transformations.base import _PanelToPanelTransformer
from source.transformations import PAA

#    TO DO: verify this returned pandas is consistent with sktime
#    definition. Timestamps?
from sktime.utils.validation.panel import check_X

# from numba import types
# from numba.experimental import jitclass


class MSAX(_PanelToPanelTransformer):
    '''
    A version of the SAX (Symbolic Aggregate approXimation) transformer called
    as Multiresolution Symbolic Aggregate approXimation (MSAX). It uses the
    algorithm SAX to return multiples transformations of the time series where
    each of them has a different set of SAX parameters.
    
    This version uses the SAX described in
    Lin, Jessica, Eamonn Keogh, Li Wei, and Stefano Lonardi.
    "Experiencing SAX: a novel symbolic representation of time series."
    Data Mining and knowledge discovery 15, no. 2 (2007): 107-144.
    and change the parameters in order to get different words depending of the
    time series set.
    
    Overview: Considering all time series with the same length, it does:
        for each series:
            for each window length:
                discretizes the series using the SAX algorithm;
                It discretization creates a bag of words whith no
                interesection between words from different bags;
        return a bag of bags.

    Parameters
    ----------
    alphabet_size : int, optional
        Defines the number of unique symbols to discretize the series.
        The default is 4.         
    remove_repeat_words: boolean, optional
        If true, equal sequential words are removed and only one is stored. 
        Otherwise all words are kept.
        The default is False.
    normalize : boolean, optional
        Defines if each window is normalized before the discretization.
        If False the breakpoints must be recalculate estimating the data
        distribution.
        The default is True.

    '''

    _tags = {"univariate-only": True, "fit-in-transform": True}

    def __init__(self,
                 alphabet_size = 4,
                 normalize = True,
                 remove_repeat_words=False):

        self.alphabet_size = alphabet_size
        self.normalize = normalize 
        self.remove_repeat_words = remove_repeat_words
        
        self._breakpoints = self._generate_breakpoints()
        self._alphabet = self._generate_alphabet()

        if self.alphabet_size < 2 or self.alphabet_size > 4:
            raise RuntimeError("Alphabet size must be an integer between 2 and 4")

        super(MSAX, self).__init__()


    def transform(self, X, window_lengths, y=None):
        """
        Parameters
        ----------
        X : pandas Series
            Time series as a Series with all time steps.
        window_lengths : set, list or nd-array
            Set of window lengths that will generate the splits within each time
            series. The numbers of subseries generated depends on the length of
            the time series and the length of the window. The normalization is
            the same as it was defined for all windows generated.
        
        Returns
        -------
        word_sequence: 
        """

        if(type(X) != pd.Series):
            raise TypeError('MSAX.transform requires as input one timeseries as a pandas Series \
                            and it received the data as {}'.format(type(X)))
        word_length = window_lengths[0]
        if word_length < 1 or word_length > 16:
            raise RuntimeError("Word length must be an integer between 1 and 16")

        # TODO 
        # another function breakpoints if normalize is False
        series_len = X.size

        word_sequences = dict()

        for window_len in window_lengths:
            
            num_windows_per_inst = series_len - window_len + 1
            split = np.array([X[i:i+window_len] for i in range(num_windows_per_inst)])

            split = scipy.stats.zscore(split, axis=1)
            split = np.nan_to_num(split)

            paa = PAA(num_intervals = word_length)
            patterns = paa.transform_univariate(split)
            
            words = pd.Series([self._create_word(pattern)
                               for pattern in patterns])
            word_sequences[window_len] = words

        return word_sequences

    def _create_word(self, pattern):
        word = ''
        for i in range(pattern.size):
            for bp in range(self.alphabet_size):
                if pattern[i] <= self._breakpoints[bp]:
                    word = word + self._alphabet[bp]
                    break
        return word

    def _add_to_bag(self, bag, word, last_word):
        if self.remove_repeat_words and word == last_word:
            return False
        bag[word] = bag.get(word, 0) + 1
        return True

    def _generate_breakpoints(self):
        # Pre-made gaussian curve breakpoints from UEA TSC codebase
        return {
            2: [0, sys.float_info.max],
            3: [-0.43, 0.43, sys.float_info.max],
            4: [-0.67, 0, 0.67, sys.float_info.max],
            5: [-0.84, -0.25, 0.25, 0.84, sys.float_info.max],
            6: [-0.97, -0.43, 0, 0.43, 0.97, sys.float_info.max],
            7: [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07, sys.float_info.max],
            8: [-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15, sys.float_info.max],
            9: [-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22, sys.float_info.max],
            10: [-1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28, sys.float_info.max]
        }[self.alphabet_size]
    
    def _generate_alphabet(self):
        # Unique alphabet symbols to be used in the discretization
        return {
            2: ['a', 'b'],
            3: ['a', 'b','c'],
            4: ['a', 'b','c','d'],
            5: ['a', 'b','c','d','e'],
            6: ['a', 'b','c','d','e','f'],
            7: ['a', 'b','c','d','e','f','g'],
            8: ['a', 'b','c','d','e','f','g','h'],
            9: ['a', 'b','c','d','e','f','g','h','i'],
            10: ['a', 'b','c','d','e','f','g','h','i','j'],
        }[self.alphabet_size]