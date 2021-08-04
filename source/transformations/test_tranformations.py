
import sys
sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace')

import pandas as pd
import numpy as np
import unittest
import scipy.stats
import random
from source.transformations.paa import PAA
from source.transformations.msax import MSAX



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
    
    def test_fit_transform(self):
        seed = random.randrange(1000)
        random.seed(seed)
        
        timeseries = pd.Series([-80, -53, -40, -67, 15, -79,
                                -79, -39, -66, 83, -90, -2])
        
        windows_lengths = np.array([9, 10])
        word_lengths = np.array([3, 4])
        
        res1 = np.array(['bcb','bbc','cbc','bbc'])
        res2 = np.array(['bcbd','bcbc','bbcc'])
        
        correct_answer = [res1,res2]
        
        msax = MSAX(alphabet_size=4)
        actual_answer = msax.transform(timeseries,windows_lengths,word_lengths)
        
        for i in range(actual_answer.size):
            comparison = correct_answer[i] == actual_answer.iloc[i]
            assert comparison.all() == True

        
        
        
        
        
        
        
        
        
        
        
        
        