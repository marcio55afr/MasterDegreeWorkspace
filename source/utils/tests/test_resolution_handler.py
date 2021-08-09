import sys
sys.path.append('C:\\Users\\marci\\Desktop\\MasterDegreeWorkspace\\source')

import pandas as pd
import numpy as np
import unittest
import random
from utils.resolution_handler import ResolutionHandler


class Test_function_get_info_from(unittest.TestCase):
    
    def test_WindowWordNgram_from_NgramResolution(self):
        ngram_resolution = '55 44 8'
        
        expected_window = 55
        expected_word = 44
        expected_ngram = 8
        
        received_window = ResolutionHandler.get_window_from(ngram_resolution)
        received_word = ResolutionHandler.get_word_from(ngram_resolution)
        received_ngram = ResolutionHandler.get_ngram_from(ngram_resolution)
        
        assert expected_window == received_window
        assert expected_word == received_word
        assert expected_ngram == received_ngram


class Test_function_generate_window_lengths(unittest.TestCase):
    
    def test_isunique(self):
        smallest_window = 4
        biggest_window = 32
        num_windows_list = np.arange(8) + 2
        
        expected_isunique = True
        
        for num_windows in num_windows_list:
            received_windows = ResolutionHandler.generate_window_lengths(smallest_window,
                                                                     biggest_window,
                                                                     num_windows)
            received_windows_isunique = len(received_windows) == len(set(received_windows))
            assert expected_isunique == received_windows_isunique
    
    def test_SmallestBiggestWindow_ispresent(self):
        smallest_window = 4
        biggest_window = 32
        num_windows_list = np.arange(biggest_window-smallest_window-2) + 2
        
        expected_smallest_window_ispresent = True
        expected_biggest_window_ispresent = True
        
        for num_windows in num_windows_list:
            received_windows = ResolutionHandler.generate_window_lengths(smallest_window,
                                                                     biggest_window,
                                                                     num_windows)
            received_smallest_window_ispresent = smallest_window in received_windows
            received_biggest_window_ispresent = biggest_window in received_windows
            assert expected_smallest_window_ispresent == received_smallest_window_ispresent
            assert expected_biggest_window_ispresent == received_biggest_window_ispresent
  

class Test_function_generate_word_lengths(unittest.TestCase):
    
    def test_isunique(self):
        smallest_window = 4
        biggest_window = 32
        num_windows_list = np.arange(biggest_window-smallest_window-2) + 2
        smallest_word = 2
        dimension_reduction_list = np.arange(.1,1.1,.1)
        
        expected_isunique = True
        
        for num_windows in num_windows_list:
            windows = ResolutionHandler.generate_window_lengths(smallest_window,
                                                                     biggest_window,
                                                                     num_windows)
            for dimension_reduction in dimension_reduction_list:
                received_words = ResolutionHandler.generate_word_lengths(windows,
                                                                     smallest_word,
                                                                     dimension_reduction)
                received_words_isunique = len(received_words) == len(set(received_words))
                assert expected_isunique == received_words_isunique
        
class Test_function_generate_dataframe(unittest.TestCase):
    
    def test_NoRises(self):
        smallest_window = 4
        biggest_window = 32
        num_windows = 10
        smallest_word = 2
        dimension_reduction = .8
        windows = ResolutionHandler.generate_window_lengths(smallest_window,
                                                                 biggest_window,
                                                                 num_windows)
        ResolutionHandler.generate_word_lengths(windows,
                                            smallest_word,
                                            dimension_reduction)
        assert True
  
    def test_NoneNgramBigger_than_BiggestWindowInColumn(self):
        smallest_window = 4
        biggest_window = 32
        num_windows_list = np.arange(biggest_window-smallest_window-2) + 2
        smallest_word = 2
        dimension_reduction_list = np.arange(.1,1.1,.1)
        max_ngram = round(biggest_window/smallest_window)
        
        expected_ngramIs_ValidAndBigger = False
        
        for num_windows in num_windows_list:
            windows = ResolutionHandler.generate_window_lengths(smallest_window,
                                                                     biggest_window,
                                                                     num_windows)
            for dimension_reduction in dimension_reduction_list:
                received_words = ResolutionHandler.generate_word_lengths(windows,
                                                                     smallest_word,
                                                                     dimension_reduction)
                matrix = ResolutionHandler.generate_resolution_matrix(windows, received_words, max_ngram)
                
                for i in range(matrix.shape[0]):
                    for j in range(matrix.shape[1]):
                        window = ResolutionHandler.get_window_from(matrix.columns[j])
                        ngram_isbigger = (i*window) > biggest_window
                        ngram_valid = matrix.iloc[i,j]
                        assert (expected_ngramIs_ValidAndBigger == (ngram_valid and ngram_isbigger))
                






















