import sys

sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace/source')
sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace')

import pandas as pd
import numpy as np
from sktime.classification.base import BaseClassifier
from sklearn.linear_model import LogisticRegression

from source.utils import ResolutionMatrix
from sktime.transformations.panel.dictionary_based import SFA, SAX
from sklearn.model_selection import cross_validate

from sklearn.feature_selection import chi2


class SearchTechnique_CV_RSFS(BaseClassifier):
    """
        Cross-validation approach with Random Selection before the 
        Feature Selection
    
    """

    def __init__(self,
                 word_length=6,
                 alphabet_size=4,
                 # word_ranking_method = 'chi2',
                 # word_selection = 'best n words', # ['p threshold', 'best n words']
                 # p_threshold = 0.05,
                 discretization='SFA',
                 max_window_length=.5,
                 max_num_windows=20,
                 remove_repeat_words=False,
                 n_words=None,
                 normalize=True,
                 verbose=False,
                 random_state=None):

        # if (word_selection != 'p threshold') and (word_selection != 'best n words'):
        #    raise ValueError('The word selection must the a valid method of selection, as "p threshold" or "best n words"')

        self.word_length = word_length
        self.alphabet_size = alphabet_size
        self.max_num_windows = max_num_windows
        self.max_window_length = max_window_length
        self.remove_repeat_words = remove_repeat_words

        # self.p_threshold = p_threshold
        self.n_words = n_words
        self.normalize = normalize
        self.verbose = verbose

        self.random_state = random_state

        self.discretization = discretization
        self.discretizers = pd.Series()

        self.clf = LogisticRegression(max_iter=5000,
                                      random_state=random_state)

        self.ts_length = None
        self.windows = None
        self.results = pd.DataFrame()
        self.selected_words = set()

    def _fit(self, data, labels):

        if type(data) != pd.DataFrame:
            raise TypeError("The data must be a type of pd.DataFrame."
                            " It was received as {}".format(type(data)))

        if type(labels) != pd.Series:
            raise TypeError("The data must be a type of pd.Series."
                            " It was received as {}".format(type(labels)))

        if data.shape[0] != labels.shape[0]:
            raise RuntimeError('The labels isn\'t compatible with the data received')

        if labels is not None:
            if (data.index.values != labels.index.values).any():
                raise RuntimeError("The samples indices must be the equal to "
                                   "the labels indices")

        self.ts_length = data.iloc[0, 0].size
        self.windows = ResolutionMatrix(self.ts_length,
                                        self.word_length,
                                        self.max_window_length,
                                        self.max_num_windows).matrix.columns.values

        if self.verbose:
            print('\nFitting the Classifier with data...')

            print('\nFitting the transformers...')
            for w in self.windows:
                print('_', end='')
            print('')
        for window in self.windows:
            if self.verbose:
                print('#', end='')

            if self.discretization == 'SFA':
                disc = SFA(window_size=window,
                           word_length=self.word_length,
                           alphabet_size=self.alphabet_size,
                           norm=self.normalize,
                           remove_repeat_words=self.remove_repeat_words,
                           return_pandas_data_series=True,
                           n_jobs=-1
                           ).fit(data, labels)
            else:
                disc = SAX(window_size=window,
                           word_length=self.word_length,
                           alphabet_size=self.alphabet_size,
                           remove_repeat_words=self.remove_repeat_words,
                           return_pandas_data_series=True
                           ).fit(data, labels)

            self.discretizers.loc[window] = disc

        if self.verbose:
            print('\nSearching for the best CV...')
            for w in self.windows:
                print('_', end='')
            print('')
        for window in self.windows:
            if self.verbose:
                print('#', end='')
            disc = self.discretizers[window]
            word_sequence = disc.transform(data, labels)
            bag_of_words = self._get_feature_matrix(word_sequence)

            if (self.n_words is None) or (self.n_words <= 0):
                raise ValueError('When feature_selection is selected as True '
                                 'the n_words must be a positive number.')
            bag_of_words = self._feature_selection(bag_of_words, labels)

            results = cross_validate(self.clf,
                                     bag_of_words,
                                     labels,
                                     cv=10,
                                     scoring=['roc_auc_ovo']
                                     )
            for score in results.keys():
                self.results.loc[score, window] = results[score].mean()

        best_window = 0
        best_r = 0
        for window in self.windows:
            r = self.results.loc['test_roc_auc_ovo', window]
            if r > best_r:
                best_window = window
                best_r = r

        if self.verbose:
            print('\nTraining the classifier with the best CV...')
        self.windows = best_window
        self.discretizers = self.discretizers[window]

        word_sequence = self.discretizers.transform(data)
        bag_of_words = self._get_feature_matrix(word_sequence)

        bag_of_words = self._feature_selection(bag_of_words, labels)

        self.selected_words = bag_of_words.columns.values
        self.clf.fit(bag_of_words, labels)
        self.classes_ = sorted(self.clf.classes_)
        self._is_fitted = True

    def _predict(self, data):

        if self.verbose:
            print('Predicting data with the Classifier...\n')

        self.check_is_fitted()

        word_sequence = self.discretizers.transform(data)
        bag_of_words = self._get_feature_matrix(word_sequence)
        bag_of_words = self._feature_filtering(bag_of_words)

        return self.clf.predict(bag_of_words)

    def _predict_proba(self, data):

        if self.verbose:
            print('Predicting data with the Classifier...\n')

        self.check_is_fitted()

        word_sequence = self.discretizers.transform(data)
        bag_of_words = self._get_feature_matrix(word_sequence)
        bag_of_words = self._feature_filtering(bag_of_words)

        return self.clf.predict_proba(bag_of_words)

    def _feature_selection(self, bag_of_words, labels):

        bag_of_words = bag_of_words.sample(axis=1,
                                           frac=0.5,
                                           random_state=self.random_state)

        rank_value, p = chi2(bag_of_words, labels)
        word_rank = pd.DataFrame(index=bag_of_words.columns)
        word_rank['rank'] = rank_value
        word_rank = word_rank.sort_values('rank', ascending=False)
        best_words = word_rank.iloc[0:self.n_words].index.values

        return bag_of_words[best_words]

    def _feature_filtering(self, bag_of_words):

        indices = bag_of_words.columns.get_indexer(self.selected_words)

        mask = indices >= 0
        intersecting_words = self.selected_words[mask]
        bag_of_words = bag_of_words[intersecting_words]
        for missing_word in self.selected_words[~mask]:
            bag_of_words.insert(bag_of_words.shape[1], missing_word, 0)

        if self.verbose:
            print('Intersecting words: {}'.format(mask.sum()))
        return bag_of_words[self.selected_words]

    def _get_feature_matrix(self, word_sequence):

        if self.discretization == 'SAX':
            word_counting = word_sequence[0].map(pd.value_counts)
            feature_matrix = pd.concat(list(word_counting), axis=1).T.fillna(0).astype(np.int16)
        else:
            feature_matrix = pd.concat(word_sequence[0].tolist(), axis=1).T.fillna(0).astype(np.int16)
        return feature_matrix
