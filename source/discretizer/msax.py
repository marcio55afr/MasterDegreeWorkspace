# -*- coding: utf-8 -*-
import sys

import numpy as np
import pandas as pd
import scipy.stats

from sktime.transformations.base import _PanelToPanelTransformer
from sktime.transformations.panel.dictionary_based import PAA

#    TO DO: verify this returned pandas is consistent with sktime
#    definition. Timestamps?
from sktime.utils.validation.panel import check_X

# from numba import types
# from numba.experimental import jitclass

__author__ = "Matthew Middlehurst"


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
                 word_prop = .80,
                 normalize = True,
                 remove_repeat_words=False):

        self.alphabet_size = alphabet_size
        self.word_prop = word_prop
        self.normalize = normalize 
        self.remove_repeat_words = remove_repeat_words

        if self.alphabet_size < 2 or self.alphabet_size > 4:
            raise RuntimeError("Alphabet size must be an integer between 2 and 4")

        super(MSAX, self).__init__()

    def transform(self, X, window_lengths, word_lengths, y=None):
        """
        Parameters
        ----------
        X : pandas DataFrame of shape [1, n_dimension]
            Time series as a Dataframe with each dimension in each cell.
        window_lengths : list or iterator
            Set of window lengths that will generate the splits within each time
            series. The numbers of subseries generated depends on the length of
            the time series and the length of the window. The normalization is
            the same as it was defined for all windows generated.
        word_lengths : list or iterator
            Set of words lengths that will generate the bag of bags. Each word
            length generates each bag of words and the same alphabet is kept for
            all of the discretizations.
        
        Returns
        -------
        dims: Pandas data frame with first dimension in column zero
        """
        
        
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)
        X = X.squeeze(1)

        for word_length in word_lengths:
            if word_length < 1 or word_length > 16:
                raise RuntimeError("Every word length must be an integer between 1 and 16")

        # TODO 
        # another function breakpoints if normalize is False
        breakpoints = self._generate_breakpoints()
        series_length = X.shape[1]

        bags = pd.DataFrame()
        dim = []

        for window_length in range(window_lengths):
            bag = {}
            lastWord = -1

            num_windows_per_inst = series_length - window_length + 1
            split = np.array(
                X[
                    np.arange(window_length)[None, :]
                    + np.arange(num_windows_per_inst)[:, None],
                ]
            )

            split = scipy.stats.zscore(split, axis=1)

            paa = PAA(num_intervals=self.word_length)
            patterns = paa.fit().transform_univariate(split)
            patterns = np.asarray([a.values for a in patterns.iloc[:, 0]])

            for n in range(patterns.shape[0]):
                pattern = patterns[n, :]
                word = self._create_word(pattern, breakpoints)
                lastWord = self._add_to_bag(bag, word, lastWord)

            dim.append(pd.Series(bag) if self.return_pandas_data_series else bag)

        bags[0] = dim

        return bags

    def _create_word(self, pattern, breakpoints):
        word = 0
        for i in range(self.word_length):
            for bp in range(self.alphabet_size):
                if pattern[i] <= breakpoints[bp]:
                    word = (word << 2) | bp
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
            10: [
                -1.28,
                -0.84,
                -0.52,
                -0.25,
                0.0,
                0.25,
                0.52,
                0.84,
                1.28,
                sys.float_info.max,
            ],
        }[self.alphabet_size]