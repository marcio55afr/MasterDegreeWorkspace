import sys
sys.path.append('C:\\Users\\danie\\Documents\\Marcio\\MasterDegreeWorkspace\\source')

import pandas as pd
import numpy as np
import unittest
import scipy.stats
import random
from transformations.paa import PAA
from transformations.msax import MSAX



class TestPAAaprroximation(unittest.TestCase):
    
    def test_transform_univariate(self):
        timeseries = np.asarray([
            list(range(9)),
            [3]*9,
            [-9,-9,-9,0,0,0,0,50,100]
        ])
        
        word_length = 3
        correct_answer = np.asarray([
            [1,4,7],
            [3,3,3],
            [-9,0,50]
        ])
        
        paa = PAA(num_intervals=word_length)
        approximations = paa.fit(None).transform_univariate(timeseries)
        
        print(approximations)
        
        comparison = approximations == correct_answer
        assert comparison.all() == True

class TestMSAXdiscretization(unittest.TestCase):
    
    def test_transform_invalid_param_x(self):
        timeseries = pd.DataFrame([-80, -53, -40, -67, 15, -79,
                                -79, -39, -66, 83, -90, -2])
        msax = MSAX()
        with self.assertRaises(TypeError):
            msax.transform(timeseries,None,None)

    def test_transform_invalid_param_windows(self):

        timeseries = pd.Series([-80, -53, -40, -67, 15, -79,
                                -79, -39, -66, 83, -90, -2])
        window_lengths = [9, 10, 5]
        word_lengths = [3, 4]        
        msax = MSAX()
        with self.assertRaises(RuntimeError):
            msax.transform(timeseries,window_lengths,word_lengths)

    def test_transform_invalid_param_words(self):

        timeseries = pd.Series([-80, -53, -40, -67, 15, -79,
                                -79, -39, -66, 83, -90, -2])
        window_lengths = [9, 8, 5]
        word_lengths = '985'
        msax = MSAX()
        with self.assertRaises(TypeError):
            msax.transform(timeseries,window_lengths,word_lengths)
            
    def test_transform_valid_params(self):
        timeseries = pd.Series([-80, -53, -40, -67, 15, -79,
                                -79, -39, -66, 83, -90, -2])        
        a = np.array([3])
        msax = MSAX()
        msax.transform(timeseries,a,a)
        assert True == True, 'Expect no exceptions'


    def test_transform(self):
        seed = random.randrange(1000)
        random.seed(seed)
        
        timeseries = pd.Series([-80, -53, -40, -67, 15, -79,
                                -79, -39, -66, 83, -90, -2])
        
        window_lengths = [9, 10]
        word_lengths = [3, 4]
        
        res1 = ['bcb','bbc','cbc','bbc']
        res2 = ['bcbd','bcbc','bbcc']
        
        correct_answer = {'9 3':res1,
                          '10 4':res2}
        
        msax = MSAX(alphabet_size=4)
        actual_answer = msax.transform(timeseries,window_lengths,word_lengths)
        
        for key in actual_answer.keys():
            comparison = actual_answer[key] == correct_answer.get(key,[])
            assert comparison == True

        
        
        
        
        
        
        
        
        
        
        
        
        