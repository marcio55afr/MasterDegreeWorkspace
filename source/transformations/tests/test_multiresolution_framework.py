

import unittest
from source.transformations import MultiresolutionFramework
from source.experiments.database import get_train_test_split, DATASET_NAMES
from source.utils import ResolutionMatrix


#class TestFunction_fit_transform(unittest.TestCase):
    

#    def test_raise_no_errors(self):
word_length = 6
train, labels = get_train_test_split(DATASET_NAMES[0], split='train')
ts_len = train.iloc[0,0].size

rm = ResolutionMatrix(ts_len, word_length)

transformer = MultiresolutionFramework(rm.matrix, word_len = word_length)

word_sequences = transformer.fit_transform(train, labels)
        
        
# -*- coding: utf-8 -*-

