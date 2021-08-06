import pandas as pd
import numpy as np
import unittest
import random
from utils.ngram_extractor import NgramExtractor

class TestFunction_get_bob(unittest.TestCase):

    def test_simple_run(self):
        word_sequence = {'2 1' : pd.Series(['a','b','c','d','e'])}
        reso_matrix = pd.DataFrame([True], index=[1], columns=['2 1'])

        NgramExtractor.get_bob(word_sequence, reso_matrix)
    
    def test_return_type_and_columns(self):
        word_sequence = {'2 1' : pd.Series(['a','b','c','d','e']),
                         '3 2' : pd.Series(['aa','bb','cc','dd','ee'])}
        reso_matrix = pd.DataFrame(True, index=[1], columns=['2 1','3 2'])

        expected_type = pd.DataFrame
        expected_columns = ['ngram word','resolution','ngram']

        result = NgramExtractor.get_bob(word_sequence, reso_matrix)
        
        received_type = type(result)
        received_columns = result.columns

        assert expected_type == received_type
        for col in expected_columns:
            assert (col in received_columns)


class TestFunction_get_bonw(unittest.TestCase):
    
    def test_simple_run(self):
        sequence = ['t','e','s','t','r','u','n']
        window_length = 2
        valid_ngrams = [1]

        NgramExtractor.get_bonw(pd.Series(sequence), window_length, valid_ngrams)

    def test_return_bonw_columns(self):
        sequence = ['t','e','s','t','r','u','n']
        window_length = 2
        valid_ngrams = [1]

        expected_columns = ['ngram word', 'frequency']

        df_bonw = NgramExtractor.get_bonw(pd.Series(sequence), window_length, valid_ngrams)

        for col in expected_columns:
            assert (col in df_bonw.columns)

    def test_return_all_ngrams(self):
        sequence = ['a','b','bigram','window_intersection','test','c','d','e']
        window_length = 2
        valid_ngrams = [1,2]
        
        correct_ngram_words = sequence + [
            'a bigram', 'b window_intersection', 'bigram test',
            'window_intersection c', 'test d', 'c e'
        ]
        
        df_bonw = NgramExtractor.get_bonw(pd.Series(sequence), window_length, valid_ngrams)
        received_ngram_words = df_bonw['ngram word'].values
        
        comparison = received_ngram_words == correct_ngram_words
        assert comparison.all()
        
    def test_return_no_overlaying_ngrams(self):
        sequence = ['a','b','bigram','window_intersection','test','c','d','e',
                    'ThisShould','NotHappen','test','more','words']
        window_length = 2
        valid_ngrams = [1,2]
        
        incorrect_ngram = 'ThisShould NotHappen'
        
        df_bonw = NgramExtractor.get_bonw(pd.Series(sequence), window_length, valid_ngrams)
        received_ngram_words = df_bonw['ngram word'].values
        
        assert (incorrect_ngram in received_ngram_words) == False
    
    def test_ngram_windowlen(self):
        sequence = ['a','b','ad','da','test','c','d','e','f','This','h','i','j',
                    'Should','h','kk','t','Happen','test','more','words',
                    'For','b','ad','da','Sure','c','d','e','f','g','h','i','j']
        window_length = 4
        valid_ngrams = [1,2,3,5]
        
        three_gram_word = 'This Should Happen'
        five_gram_word = 'This Should Happen For Sure'
        
        df_bonw = NgramExtractor.get_bonw(pd.Series(sequence), window_length, valid_ngrams)
        received_ngram_words = df_bonw['ngram word'].values
        
        assert (three_gram_word in received_ngram_words)
        assert (five_gram_word in received_ngram_words)

    def test_ngram_word_unique(self):
        sequence = ['a','b','ad','da','test','c','d','e','f','This','h','i','j',
                    'b','ad','da','..','c','d','he','hgf','fg','gh','di','ja',
                    'hui','oj','random','l','e','t','t','e','r','s','and','so',
                    'on', 'b','ad','da','..','c','d','e','f','g','h','i','j']
        window_length = 3
        valid_ngrams = [1,2,3,5,6]
        
        expected_unique_words = True        

        df_bonw = NgramExtractor.get_bonw(pd.Series(sequence), window_length, valid_ngrams)
        received_bonw = df_bonw.set_index('ngram word')['frequency']
        
        assert df_bonw['ngram word'].is_unique == expected_unique_words

    def test_ngram_count(self):
        sequence = ['a','b','ad','da','test','c','d','e','f','This','h','i','j',
                    '.........................................................',
                    'this','this','this','this','this','this','this','this',
                    'appears','8','times', 'and', 'this_this', 'will', 'appear',
                    '5','times','so','lets','test','that',
                    '.........................................................',
                    '....','b','ad','da','..','c','d','e','f','g','h','i','j']
        window_length = 3
        valid_ngrams = [1,2,3]
        
        expected_count_word = {'this': 8,
                               'this this': 5}

        df_bonw = NgramExtractor.get_bonw(pd.Series(sequence), window_length, valid_ngrams)
        received_bonw = df_bonw.set_index('ngram word')['frequency']
        
        assert received_bonw['this'] == expected_count_word['this']
        assert received_bonw['this this'] == expected_count_word['this this']