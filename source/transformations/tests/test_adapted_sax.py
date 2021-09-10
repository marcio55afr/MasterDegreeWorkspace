

import unittest
from source.transformations import AdaptedSAX
from source.experiments.database import get_Xy_from, DATASET_NAMES



class TestFunction_fit_transform(unittest.TestCase):
    

    def test_raise_no_errors(self):
        train, labels = get_Xy_from(DATASET_NAMES[0], split='train')        

        transformer = AdaptedSAX(
            word_length = 4,
            return_pandas_data_series=False)
        transformer.fit_transform(train, labels)
        
        
        
        transformer = AdaptedSAX(
            word_length = 4,
            return_pandas_data_series=True)
        transformer.fit_transform(train, labels)
