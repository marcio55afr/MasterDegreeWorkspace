# -*- coding: utf-8 -*-

from source.transformations import AdaptedSFA
import unittest

from source.experiments.database import get_train_test_split, DATASET_NAMES



class TestFunction_fit_transform(unittest.TestCase):
    

    def test_raise_no_errors(self):
        train, labels = get_train_test_split(DATASET_NAMES[0], split='train')
        
        transformer = AdaptedSFA(
            word_length = 4,
            return_pandas_data_series=False)
        transformer.fit_transform(train, labels)
        
        transformer = AdaptedSFA(
            word_length = 4,
            return_pandas_data_series=True)
        transformer.fit_transform(train, labels)
        
        assert True

train, labels = get_train_test_split(DATASET_NAMES[0], split='train')

transformer = AdaptedSFA(
    word_length = 4,
    return_pandas_data_series=False)
transformer.fit_transform(train, labels)

transformer = AdaptedSFA(
    word_length = 4,
    return_pandas_data_series=True)
transformer.fit_transform(train, labels)
        