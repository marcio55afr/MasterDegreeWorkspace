# -*- coding: utf-8 -*-

from source.transformations import AdaptedSFA
import unittest
import itertools
import numpy as np
from source.utils import DatasetHandler


class TestFunction_fit_transform(unittest.TestCase):

    def test_extreme_params_SFA(self):
        df_name = DatasetHandler.smallest_dataset
        train_x, train_y, test_x, test_y = DatasetHandler.get_split_sample(10, df_name)
        ts_length = len(train_x[0])  # 15

        windows = [1, 3, 7, 15, 20]
        word_lengths = [1, 7, 15, 16]
        alphabet_sizes = [1, 2, 4, 6, 7]
        normalizes = [True, False]
        binnings = ["equi-depth", "equi-width", "information-gain", "kmeans"]
        anovas = [True, False]
        removes = [True, False]
        params = [windows, word_lengths, alphabet_sizes, normalizes, binnings, anovas, removes]

        for window, word_length, alphabet_size, normalize, binning, \
            anova, remove_rep_words in itertools.product(*params):
            # assert the initialization
            exception_raised = False
            exception_msg = ''
            try:
                sfa = AdaptedSFA(window_size=window,
                                 word_length=word_length,
                                 alphabet_size=alphabet_size,
                                 norm=normalize,
                                 binning_method=binning,
                                 anova=anova,
                                 remove_repeat_words=remove_rep_words,
                                 levels=1,
                                 return_pandas_data_series=False,
                                 n_jobs=-1)
            except Exception as e:
                exception_raised = True
                exception_msg = e
            finally:
                if ((word_length > window) or (alphabet_size > 6) or
                        (word_length < 1) or (window < 3) or (alphabet_size < 2)):
                    assert exception_raised, 'Not raised exception with wrong initial condition'
                else:
                    assert not exception_raised, \
                        f'Exception raised with correct initial condition: {exception_msg}'
            if exception_raised:
                continue

            # assert the fitting process
            exception_raised = False
            try:
                sfa = sfa.fit(train_x, train_y)
            except Exception as e:
                exception_raised = True
                exception_msg = e
            finally:
                if (word_length > ts_length) or (window > ts_length):
                    assert exception_raised, \
                        f'Not raised exception when time series length ({ts_length}) is lesser than ' + \
                        f'word length ({word_length}) or window sizes ({window})'
                else:
                    assert not exception_raised, f'Exception raised with good conditions: {exception_msg}'
            if exception_raised:
                continue

            # assert the transform process and the resultant bag of words
            bow = sfa.transform(train_x, train_y)
            lens = np.asarray([len(seq) for seq in bow])
            assert all(lens > 0), 'SFA transformation returned no words for a given series'

    def test_reproducibility(self):
        small_dataset = DatasetHandler.smallest_dataset
        x, y = DatasetHandler.get_train_data(small_dataset)

        sfa_1 = AdaptedSFA()  # parameters default
        sfa_1 = sfa_1.fit(x, y)
        transformation_1 = sfa_1.transform(x)

        transformation_1_rep = sfa_1.transform(x)

        sfa_2 = AdaptedSFA()  # parameters default
        sfa_2 = sfa_2.fit(x, y)
        transformation_2 = sfa_2.transform(x)

        assert transformation_1 == transformation_1_rep
        assert transformation_1 == transformation_2

    def test_batch_fit_transform(self):
        small_dataset = DatasetHandler.smallest_dataset
        x, y = DatasetHandler.get_train_data(small_dataset)

        sfa = AdaptedSFA()  # parameters default
        sfa = sfa.fit(x, y)
        ground_transformation = sfa.transform(x)

        # transforming in batch
        sfa = AdaptedSFA()  # parameters default
        sfa = sfa.fit(x, y)
        batch = 50
        i = 0
        batch_transformation = [[]]
        while i < len(x):
            batch_transformation[0] += sfa.transform(x[i:i+batch])[0]
            i += batch

        # fitting and transforming in batch
        sfa = AdaptedSFA()  # parameters default
        batch = 50
        i = 0
        while i < len(x):
            sfa = sfa.fit(x[i:i+batch], y[i:i+batch])
            i += batch

        i = 0
        batch_fit_transformation = [[]]
        while i < len(x):
            batch_fit_transformation[0] += sfa.transform(x[i:i+batch])[0]
            i += batch

        assert ground_transformation == batch_transformation, 'SFA is not capable of transforming in batch'
        #assert ground_transformation == batch_fit_transformation, 'SFA is not capable of fit-transforming in batch'



    def test_fitting_large_dataset(self):
        print('Reading data')
        x, y = DatasetHandler.get_train_data('HandOutlines')

        sfa = AdaptedSFA()  # parameters default
        print('Fitting SFA transformer')
        sfa.fit(x, y)

        assert True
