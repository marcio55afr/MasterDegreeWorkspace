

import pandas as pd
import numpy as np
import unittest
from sktime.transformations.panel.dictionary_based import PAA



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
        approximations = paa.fit(timeseries).transform_univariate(timeseries)
        
        comparison = approximations == correct_answer
        self.assertTrue( comparison.all() )


t = TestPAAaprroximation()

t.test_transform_univariate()