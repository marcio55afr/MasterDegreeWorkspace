import unittest

import numpy as np

from source.transformations import AdaptedSAX
from source.utils import DatasetHandler


class TestFunction_fit_transform(unittest.TestCase):

    def test_extreme_params_SAX(self):
        df_name = DatasetHandler.smallest_dataset
        train_x, train_y, test_x, test_y = DatasetHandler.get_split_sample(10, df_name)

        ts_length = len(train_x[0])  # 15
        for window in [1, 7, 15, 20]:
            for word_length in [1, 7, 15, 16]:
                for alphabet_size in [1, 2, 4, 6, 7]:
                    for remove_rep_words in [True, False]:
                        # assert the initialization
                        exception_raised = False
                        exception_msg = ''
                        try:
                            sax = AdaptedSAX(window_size=window,
                                             word_length=word_length,
                                             alphabet_size=alphabet_size,
                                             remove_repeat_words=remove_rep_words,
                                             return_pandas_data_series=False)
                        except Exception as e:
                            exception_raised = True
                            exception_msg = e
                        finally:
                            if((word_length > window) or (alphabet_size > 6) or
                                    (window < 1) or (word_length < 1) or (alphabet_size < 2)):
                                assert exception_raised, 'Not raised exception with wrong initial condition'
                            else:
                                assert not exception_raised,\
                                    f'Exception raised with correct initial condition: {exception_msg}'
                        if exception_raised:
                            continue

                        # assert the fit-transform process
                        exception_raised = False
                        bow = []
                        try:
                            sax = sax.fit(train_x, train_y)
                            bow = sax.transform(train_x, train_y)
                        except Exception as e:
                            exception_raised = True
                            exception_msg = e
                        finally:
                            if (word_length > ts_length) or (window > ts_length):
                                assert exception_raised,\
                                    f'Not raised exception when time series length ({ts_length}) is lesser than ' +\
                                    f'word length ({word_length}) or window sizes ({window})'
                            else:
                                assert not exception_raised, f'Exception raised with good conditions: {exception_msg}'
                        if exception_raised:
                            continue

                        # assert resultant bag of words
                        lens = np.asarray([len(seq) for seq in bow])
                        assert all(lens > 0), 'Sax transformation returned no words for a given series'
