
import sys
sys.path.append('C:\\Users\\marci\\Desktop\\MasterDegreeWorkspace\\source')
import pandas as pd
import numpy as np
import unittest
import math
from utils import ResolutionMatrix, ResolutionHandler


class Test_function_init(unittest.TestCase):
    
    def test_NoRises_with_DefaultParameters(self):
        
        # default parameters
        smallest_window = 4 
        num_window = 10
        biggest_window_prop = 1/2
        # time series length must be greater than the sum of smallest window
        # and the number of windows defined
        min_ts_length = int((smallest_window + num_window)/biggest_window_prop)
        for ts_length in np.arange(50)+min_ts_length:
            ResolutionMatrix(ts_length)
        
    def test_RiseValueError_when_TslengthIsTooSmall(self):
        
        # default parameters
        smallest_window = 4 
        num_window = 10
        biggest_window_prop = 1/2
        # Any time series length smaller than 14 must rise an error
        min_ts_length = int((smallest_window + num_window)/biggest_window_prop)
        for ts_length in range(min_ts_length):
            print(min_ts_length)
            with self.assertRaises(ValueError):
                ResolutionMatrix(ts_length)

    def test_NoneNgramBigger_than_BiggestWindowInColumn(self):
        # default parameters
        smallest_window = 4 
        num_window = 10
        biggest_window_prop = 1/2
        
        # expectation
        expected_ngramIs_ValidAndBigger = False

        # test various inserts
        min_ts_length = int((smallest_window + num_window)/biggest_window_prop)
        for ts_length in np.arange(50)+min_ts_length:
            matrix = ResolutionMatrix(ts_length).matrix
            biggest_window = ResolutionHandler.get_window_from(matrix.columns[-1])
            # test every cell of the matrix
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    window = ResolutionHandler.get_window_from(matrix.columns[j])
                    ngram_isbigger = (i*window) > biggest_window
                    ngram_valid = matrix.iloc[i,j]
                    assert (expected_ngramIs_ValidAndBigger == (ngram_valid and ngram_isbigger))













