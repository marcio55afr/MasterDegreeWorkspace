import pandas as pd
import numpy as np
import unittest
import random
from source.classifiers import Classifier3M
from source.utils.handlers.dataset_handler import DatasetHandler
import itertools


class TestExtremeParameters(unittest.TestCase):

    def test_fit_transform_function(self):
        df_name = DatasetHandler.smallest_dataset
        train_x, train_y, test_x, test_y = DatasetHandler.get_split_sample(10, df_name)
        ts_length = len(train_x[0])  # 15

        ngrams = [0, 3, 10]
        word_lengths = [1, 7, 20]
        alphabet_sizes = [0, 4, 7]
        max_window_lengths = np.linspace(-1, 2, 3)
        n_sfa_resolutionss = [0, 5, 20]
        n_sax_resolutionss = [0, 5, 20]
        sfa_features_percentiles = [0, 30, 100, 200]
        sax_features_percentiles = [0, 30, 100, 200]
        random_selections = [True, False]
        normalizes = [True, False]
        params = [ngrams, word_lengths, alphabet_sizes, max_window_lengths, n_sfa_resolutionss, n_sax_resolutionss,
                  sfa_features_percentiles, sax_features_percentiles, random_selections, normalizes]

        i = 0
        for (ngram, word_length, alphabet_size, max_window_length, n_sfa_resolutions,
             n_sax_resolutions, sfa_features_percentile, sax_features_percentile,
             random_selection, normalize) in itertools.product(*params):

            # assert the initialization
            exception_raised = False
            exception_msg = ''
            try:
                clf_3m = Classifier3M(ngram=ngram,
                                      word_length=word_length,
                                      alphabet_size=alphabet_size,
                                      max_window_length=max_window_length,
                                      n_sfa_resolutions=n_sfa_resolutions,
                                      n_sax_resolutions=n_sax_resolutions,
                                      sfa_features_percentile=sfa_features_percentile,
                                      sax_features_percentile=sax_features_percentile,
                                      random_selection=random_selection,
                                      normalize=normalize,
                                      verbose=False)
            except Exception as e:
                exception_raised = True
                exception_msg = e
            finally:
                if ((ngram < 1) or (ngram > 6) or (word_length < 3) or (word_length > 16) or (alphabet_size < 2) or
                        (alphabet_size > 6) or (max_window_length <= 0) or (max_window_length > 1.0) or
                        (n_sfa_resolutions < 1) or (n_sax_resolutions < 1) or
                        (sfa_features_percentile <= 0) or (sfa_features_percentile > 100) or
                        (sax_features_percentile <= 0) or (sax_features_percentile > 100)):
                    assert exception_raised, 'Not raised exception with wrong initial condition'
                else:
                    assert not exception_raised, \
                        f'Exception raised with correct initial condition: {exception_msg}'
            if exception_raised:
                continue

            # assert the fitting process
            exception_raised = False
            try:
                clf_3m = clf_3m.fit(train_x, train_y)
                assert clf_3m._n_extracted_words > 0, 'The selected words set cannot be empty.'
            except Exception as e:
                exception_raised = True
                exception_msg = e
            finally:
                if word_length > ts_length:
                    assert exception_raised, \
                        f'Not raised exception when time series length ({ts_length}) is lesser than ' + \
                        f'word length ({word_length}))'
                else:
                    assert not exception_raised, f'Exception raised with good conditions: {exception_msg}'
            if exception_raised:
                continue

            # assert the predicting process
            predictions = clf_3m.predict(test_x)
            assert len(predictions) == len(test_y), \
                'The prediction\'s length must be the same as the test labels.'

            i += 1
            print(i)


class TestReproducibility(unittest.TestCase):

    def test_classification_rep(self):
        df_name = DatasetHandler.smallest_dataset
        x, y, test, _ = DatasetHandler.get_split_data(df_name)

        seed = 1
        clf = Classifier3M(random_state=seed)

        clf.fit(x, y)
        first_predictions = clf.predict(test)
        for _ in range(5):
            clf = Classifier3M(random_state=seed)
            clf.fit(x, y)
            assert np.array_equal(first_predictions, clf.predict(test)), \
                'Classifier3M is not reproducible'

    # with random selection
    def test_classification_rand_selection_rep(self):
        df_name = DatasetHandler.smallest_dataset
        x, y, test, _ = DatasetHandler.get_split_data(df_name)

        seed = 1
        clf = Classifier3M(random_selection=True,
                           random_state=seed)
        clf.fit(x, y)
        first_predictions = clf.predict(test)
        for _ in range(5):
            clf = Classifier3M(random_selection=True,
                               random_state=seed)
            clf.fit(x, y)
            assert np.array_equal(first_predictions, clf.predict(test)), \
                'Classifier3M with random selection is not reproducible'


class TestVariableTypes(unittest.TestCase):

    def test_predictions_type(self):
        df_name = DatasetHandler.smallest_dataset
        x, y, test, _ = DatasetHandler.get_split_data(df_name)

        clf = Classifier3M()
        clf.fit(x, y)

        predictions = clf.predict(test)
        assert type(predictions) == np.ndarray
