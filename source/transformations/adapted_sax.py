# -*- coding: utf-8 -*-
import sys
from abc import ABC

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


class AdaptedSAX(_PanelToPanelTransformer, ABC):
    """    
    A slight different version of the SAX (Symbolic Aggregate approXimation)
    Transformer, as described in Jessica Lin, Eamonn Keogh, Li Wei and
    Stefano Lonardi, "Experiencing SAX: a novel symbolic representation of 
    time series" 
    Data Mining and Knowledge Discovery, 15(2):107-144
    authored by Matthew Middlehurst.
    
    The difference here is the return, instead of return the bag of words 
    counting the frequencies, the function fit return the word sequences of 
    the tranformation of each time series in the original order. That allow to
    create ngram words and then count the frequencies.
    
    The parameters remains the same except by the one called save_words.
    Removed to save some memory.

    Parameters
    ----------
        word_length:         int, length of word to shorten window to (using
        PAA) (default 8)
        alphabet_size:       int, number of values to discretise each value
        to (default to 4)
        window_size:         int, size of window for sliding. Input series
        length for whole series transform (default to 12)
        remove_repeat_words: boolean, whether to use numerosity reduction (
        default False)
        save_words:          boolean, whether to use numerosity reduction (
        default False)

        return_pandas_data_series:          boolean, default = True
            set to true to return Pandas Series as a result of transform.
            setting to true reduces speed significantly but is required for
            automatic test.

    Attributes
    ----------
        words:      history = []

    """

    _tags = {"univariate-only": True, "fit-in-transform": True}

    def __init__(
        self,
        word_length=8,
        alphabet_size=4,
        window_size=12,
        remove_repeat_words=False,
        return_pandas_data_series=True,
    ):

        if word_length > window_size:
            raise ValueError(f"Word length ({word_length}) can not be longer than window size ({window_size})")

        if (word_length < 1) or (word_length > 16):
            raise ValueError(f"Word length ({word_length}) must be an integer between 1 and 16")

        if (alphabet_size < 2) or (alphabet_size > 6):
            raise ValueError(f"Alphabet size ({alphabet_size}) must be a integer between 2 a 6")

        self.word_length = word_length
        self.alphabet_size = alphabet_size
        self.window_size = window_size
        self.remove_repeat_words = remove_repeat_words
        self.return_pandas_data_series = return_pandas_data_series

        super(AdaptedSAX, self).__init__()

    def transform(self, X, y=None):
        """

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, 1]
            Nested dataframe with univariate time-series in cells.

        y : Numpy.Ndarray with labels of length (n_instances), not used in this transformation

        Returns
        -------
        dims: Pandas data frame with first dimension in column zero
        """
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)
        X = X.squeeze(1)
        n_instances, series_length = X.shape

        if self.window_size > series_length:
            raise RuntimeError(f"Windows size ({self.window_size}) cannot be longer" +
                               "than series length ({series_length})")

        if self.word_length > series_length:
            raise RuntimeError(f"Word length ({self.word_length}) cannot be longer" +
                               "than series length ({series_length})")

        breakpoints = self._generate_breakpoints()

        #TODO test dim as a pandas Series
        dim = []

        for i in range(n_instances):
            lastWord = -1

            #TODO test word_sequence as a pandas Series
            word_sequence = []

            num_windows_per_inst = series_length - self.window_size + 1
            split = np.array(
                X[
                    i,
                    np.arange(self.window_size)[None, :]
                    + np.arange(num_windows_per_inst)[:, None],
                ]
            )

            split = scipy.stats.zscore(split, axis=1)

            paa = PAA(num_intervals=int(self.word_length))
            data = pd.DataFrame()
            data[0] = [pd.Series(x, dtype=np.float32) for x in split]
            patterns = paa.fit_transform(data)
            patterns = np.asarray([a.values for a in patterns.iloc[:, 0]])

            for n in range(patterns.shape[0]):
                pattern = patterns[n, :]
                word = self._create_word(pattern, breakpoints)
                lastWord = self._add_to_sequence(word_sequence, word, lastWord)

            dim.append(pd.Series(word_sequence) if self.return_pandas_data_series else word_sequence)

        bags = pd.DataFrame() if self.return_pandas_data_series else [None]
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

    def _add_to_sequence(self, word_sequence, word, last_word):
        if (not self.remove_repeat_words) or (word != last_word):
            word_sequence.append(word)
        return word

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
# -*- coding: utf-8 -*-

