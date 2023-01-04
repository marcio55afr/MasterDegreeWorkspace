# -*- coding: utf-8 -*-


import sys

sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace/source')
sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace')

import random
import pandas as pd
import numpy as np
from sktime.classification.base import BaseClassifier


class RandomClassifier(BaseClassifier):
    """
        Random classifier to compare the naiviest approach to classifier
        acording to the Class Probability of the train data.
    
    """

    def __init__(self, verbose=False, random_state=None):

        self.labels = None
        self.verbose = verbose
        self.random_state = random_state
        super(RandomClassifier, self).__init__()

    def _fit(self, data, labels):

        if self.verbose:
            print('Fitting the Classifier...\n')
        self.labels = pd.Series(labels)
        self._is_fitted = True

    def _predict(self, data):

        self.check_is_fitted()
        if self.verbose:
            print('Predicting data with the Classifier...\n')

        n_samples = data.shape[0]
        _random_state = None
        if self.random_state is not None:
            _random_state = n_samples + self.random_state
        y_pred = self.labels.sample(n_samples, replace=True, random_state=_random_state)
        return y_pred.values

    '''
    def predict_proba(self, data):

        if self.verbose:
            print('Predicting data with the Classifier...\n')

        self.check_is_fitted()

        n_samples = data.shape[0]
        _random_state = None
        if self.random_state is not None:
            _random_state = n_samples + self.random_state
        y_pred = self.labels.sample(n_samples, replace=True, random_state=_random_state)

        y_proba = pd.DataFrame(0, index=range(n_samples), columns=self.labels.unique())
        for i in range(n_samples):
            y_proba.loc[i, y_pred[i]] = 1

        return y_proba.values
    '''
