import pandas as pd
import numpy as np
import unittest
import random
from utils.ngram_extractor import NgramExtractor



class TestFunctionGetNgramFrequency(unittest.TestCase):
    
    def test_return_all_ngrams(self):
        sequence = ['a','b','bigram','window_intersection','test','c','d','e']
        window_length = 2
        valid_ngrams = [1,2]
        
        correct_ngram_words = sequence + [
            'a bigram', 'b window_intersection', 'bigram test',
            'window_intersection c', 'test d', 'c e'
        ]
        
        df_bonw = NgramExtractor.get_ngram_frequency(pd.Series(sequence), window_length, valid_ngrams)
        received_ngram_words = df_bonw['ngram word'].values
        
        comparison = received_ngram_words == correct_ngram_words
        assert comparison.all()