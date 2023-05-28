# import sys
# sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace/source')
# sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace')

import time
import sys
import warnings
import pandas as pd
import numpy as np
from sktime.classification.base import BaseClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse import hstack

from source.utils import ResolutionHandler, ResolutionMatrix
from source.transformations import AdaptedSAX, AdaptedSFA

from sklearn.feature_selection import SelectPercentile, chi2


class Classifier3M(BaseClassifier):
    """
        Multiresolution approach with the Shotgun using Random Forest as clf

    """

    def __init__(self,
                 ngram=5,
                 word_length=6,
                 alphabet_size=4,
                 max_window_length=.5,
                 n_sax_resolutions=2,
                 rate_sfa_resolutions=4,
                 random_selection=False,
                 sax_features_percentile=50,
                 sfa_features_percentile=50,
                 normalize=True,
                 verbose=False,
                 random_state=None):

        if (ngram < 1) or (ngram > 6):
            raise ValueError(f'Ngram must be a integer between 1 and 6 but it was received {ngram}.')

        if (word_length < 3) or (word_length > 16):
            raise ValueError(f"Word length ({word_length}) must be an integer between 3 and 16 according" +
                             f" to SFA transform")

        if (alphabet_size < 2) or (alphabet_size > 6):
            raise ValueError(f"Alphabet size ({alphabet_size}) must be a integer between 2 and 6")

        if (max_window_length <= 0) or (max_window_length > 1.0):
            raise ValueError(f"Maximum window length ({max_window_length}) must be a positive float between 0 and 1")

        if (rate_sfa_resolutions < 1) or (n_sax_resolutions < 1):
            raise ValueError("Neither number of sax nor rate of sfa resolutions can be lesser than 1, but it was"
                             f"received: n_sfa_resolutions: {rate_sfa_resolutions},"
                             f" n_sax_resolutions: {n_sax_resolutions}")

        if rate_sfa_resolutions > 5:
            raise ValueError("The rate of sfa resolutions over sax resolutions can not be bigger than 5,"
                             f"but it was received {rate_sfa_resolutions}")

        if (sfa_features_percentile < 1) or (sfa_features_percentile > 100) or\
                (sax_features_percentile < 1) or (sax_features_percentile > 100):
            raise ValueError(
                "Both number of sax and sfa feature percentiles must be between (0,100]! And it was received: " +
                f"sfa_features_percentile: {sfa_features_percentile}, " +
                f"sax_features_percentile: {sax_features_percentile}"
            )

        self.ngram = ngram
        self.word_length = word_length
        self.alphabet_size = alphabet_size
        self.max_window_length = max_window_length

        self.n_sax_resolutions = n_sax_resolutions
        self.sax_features_percentile = sax_features_percentile
        self.rate_sfa_resolutions = rate_sfa_resolutions
        self.n_sfa_resolutions = round(rate_sfa_resolutions * n_sax_resolutions)
        self.sfa_features_percentile = sfa_features_percentile

        self.random_selection = random_selection
        self.normalize = normalize
        self.verbose = verbose
        self.random_state = random_state

        self.clf = RandomForestClassifier(criterion="gini",
                                          n_estimators=1000,
                                          class_weight='balanced_subsample',
                                          n_jobs=-1,
                                          random_state=random_state)

        # Internal Variables
        self.ts_length = None
        self.windows = None
        self.results = pd.DataFrame()
        self.sfa_id = 0
        self.sax_id = 1
        self.remove_repeat_words = False
        self._n_extracted_words = 0

        # Attributes
        self._discretizers_id = [self.sax_id, self.sfa_id]
        self._windows = pd.Series([pd.Series(dtype=object), pd.Series(dtype=object)],
                                  index=self._discretizers_id, dtype=object)
        self._vocabularies = self._windows.copy()
        self._discretizers = self._windows.copy()
        self._n_resolutions = pd.Series([self.n_sax_resolutions, self.n_sfa_resolutions],
                                        index=self._discretizers_id)
        self._feature_percentile = pd.Series([sax_features_percentile, sfa_features_percentile],
                                             index=self._discretizers_id)

    def _fit(self, data, labels):
        if type(data) != np.ndarray:
            raise TypeError("The data must be a type of numpy.ndarray."
                            " It was received as {}".format(type(data)))

        if type(labels) != np.ndarray:
            raise TypeError("The data must be a type of numpy.ndarray."
                            " It was received as {}".format(type(labels)))

        if len(data) != len(labels):
            raise RuntimeError('The labels isn\'t compatible with the data received')

        self.ts_length = len(data[0, 0])  # Equal length for all series
        if self.word_length > self.ts_length:
            warnings.warn(f'Word length was reduced to {self.ts_length} because its previous value ({self.word_length})'
                          f' was bigger than time series length ({self.ts_length}).')
            self.word_length = self.ts_length

        if self.verbose:
            print('\nFitting the transformers...')
        for disc_id in self._discretizers_id:
            n_resolution = self._n_resolutions[disc_id]
            self._windows[disc_id] = ResolutionMatrix(self.ts_length,
                                                      self.word_length,
                                                      self.max_window_length,
                                                      n_resolution).matrix.columns.values
            if self.verbose:
                print('_'*len(self._windows[disc_id]), end='')

        if self.verbose:
            print('')

        for disc_id in self._discretizers_id:
            windows = self._windows[disc_id]
            discretizers = pd.Series(dtype=object)
            for window in windows:
                if disc_id == self.sax_id:
                    sax = AdaptedSAX(window_size=window,
                                     word_length=self.word_length,
                                     alphabet_size=self.alphabet_size,
                                     remove_repeat_words=self.remove_repeat_words,
                                     return_pandas_data_series=False
                                     ).fit(data, labels)
                    discretizers.loc[window] = sax
                else:
                    sfa = AdaptedSFA(window_size=window,
                                     word_length=self.word_length,
                                     alphabet_size=self.alphabet_size,
                                     norm=self.normalize,
                                     remove_repeat_words=self.remove_repeat_words,
                                     return_pandas_data_series=False,
                                     n_jobs=None
                                     ).fit(data, labels)
                    discretizers.loc[window] = sfa

                if self.verbose:
                    print('#', end='')
            self._discretizers[disc_id] = discretizers

        bob = self._extract_features(data, labels)
        self._n_extracted_words = 0
        for disc_id in self._discretizers_id:
            windows = self._windows[disc_id]
            vocabs = self._vocabularies[disc_id]
            for window in windows:
                self._n_extracted_words += len(vocabs[window])

        if self._n_extracted_words == 0:
            raise RuntimeError("The feature extraction process selected no words.")

        if self.verbose:
            print('training the random forest')
        self.clf = self.clf.fit(bob, labels)
        self._is_fitted = True

    def _predict(self, data):
        if self.verbose:
            print('Predicting data with the Classifier...\n')

        self.check_is_fitted()

        bag_of_bags = self._extract_features(data, None)
        bag_of_bags = bag_of_bags.asformat('csr')
        return self.clf.predict(bag_of_bags)

    def _predict_proba(self, data):
        if self.verbose:
            print('Predicting data with the Classifier...\n')

        self.check_is_fitted()

        bag_of_bags = self._extract_features(data, None)
        bag_of_bags = bag_of_bags.asformat('csr')
        return self.clf.predict_proba(bag_of_bags)

    def _extract_features(self, data, labels):
        if self.verbose:
            print('\n\nExtracting features from all resolutions...')
            print('_'*(len(self._windows[self.sax_id])+len(self._windows[self.sfa_id])), end='')
            print('')

        bob = None

        for discretizer_id in [self.sax_id, self.sfa_id]:
            windows = self._windows[discretizer_id]
            vocabularies = self._vocabularies[discretizer_id]
            discretizers = self._discretizers[discretizer_id]
            feature_percentile = self._feature_percentile[discretizer_id]

            for window in windows:
                if self.verbose:
                    print('#', end='')

                disc = discretizers[window]
                word_sequence = disc.transform(data, labels)
                ngram_sequence = self._extract_ngram_words(word_sequence)

                # Parsing the word sequences to a word frequency matrix
                if labels is not None:
                    # Fitting
                    bag_of_words, vocab = self._get_feature_sparsematrix(ngram_sequence)
                    bag_of_words, vocab = self._feature_selection(bag_of_words, vocab, labels, feature_percentile)
                    vocabularies.loc[window] = vocab
                else:
                    # Predicting
                    vocab = vocabularies.loc[window]
                    bag_of_words = self._match_feature_sparsematrix(ngram_sequence, vocab)
                    bag_of_words = self._match_feature_format(bag_of_words, vocab)

                # Concatenating bag of words of different resolutions: bag of bags (BoB)
                if bob is None:
                    bob = bag_of_words
                else:
                    bob = hstack([bob, bag_of_words], dtype=np.int16)
        return bob

    def _feature_selection(self, bag_of_words, vocabulary, labels, feature_percentile):
        if self.random_selection:
            items = sorted(vocabulary.items(), key=lambda x: x[1])
            words = list(zip(*items))[0]
            vocab_size = len(vocabulary)
            rand_selection = np.random.choice(vocab_size, size=vocab_size//2, replace=False)
            rand_selection = sorted(rand_selection)

            rand_words = np.asarray(words)[rand_selection]
            bag_of_words = bag_of_words[:, rand_selection]
            vocabulary = dict(zip(rand_words, range(bag_of_words.shape[1])))

        items = sorted(vocabulary.items(), key=lambda x: x[1])
        words = list(zip(*items))[0]

        rank_value, _ = chi2(bag_of_words, labels)
        bottom_limit = np.percentile(rank_value, [100-feature_percentile])
        feature_mask = rank_value >= bottom_limit

        selected_words = np.asarray(words)[feature_mask]
        bag_of_selected_words = bag_of_words[:, feature_mask]
        selected_vocabulary = dict(zip(selected_words, range(bag_of_selected_words.shape[1])))

        return bag_of_selected_words, selected_vocabulary

    def _match_feature_format(self, bag_of_bags: csr_matrix, vocabulary: dict):
        n_fitted_words = len(vocabulary)
        if n_fitted_words > bag_of_bags.shape[1]:
            bag_of_bags.resize(bag_of_bags.shape[0], n_fitted_words)
        return bag_of_bags

    def _get_feature_sparsematrix(self, ngram_sequences):
        indptr = [0]
        indices = []
        data = []
        vocabulary = {}
        for seq in ngram_sequences:
            for ngram_word in seq:
                index = vocabulary.setdefault(ngram_word, len(vocabulary))
                indices.append(index)
                data.append(1)
            indptr.append(len(indices))
        bag_of_words = csr_matrix((data, indices, indptr), dtype=int).asformat('csc')
        return bag_of_words, vocabulary
        # ngram_counts = list(map(pd.value_counts, ngram_sequences))
        # print('ngram_counts, ', sys.getsizeof(ngram_counts))
        # dfbag_of_words = pd.concat(ngram_counts, axis=1).T.fillna(0).astype(np.int32)
        # print('dfbag_of_words, ', sys.getsizeof(dfbag_of_words))
        # print(bag_of_words.toarray() == dfbag_of_words.toarray())
        # return bag_of_words

    def _match_feature_sparsematrix(self, ngram_sequences, vocabulary):
        indptr = [0]
        indices = []
        data = []
        for seq in ngram_sequences:
            for ngram_word in seq:
                index = vocabulary.get(ngram_word)
                if index:
                    indices.append(index)
                    data.append(1)
            indptr.append(len(indices))
        if data:
            bag_of_words = csr_matrix((data, indices, indptr), dtype=int).asformat('csc')
        else:
            bag_of_words = csr_matrix(([0], [len(vocabulary)-1], indptr), dtype=int).asformat('csc')
        return bag_of_words

    def _extract_ngram_words(self, word_sequences):

        def get_ngrams_from(sample_sequence):
            sample_sequence = list(map(str, sample_sequence))
            ngrams = []
            for n in range(1, self.ngram):
                n += 1
                for i in range(n):
                    ngrams += zip(*[iter(sample_sequence[i:])] * n)

            return sample_sequence + list(map(' '.join, ngrams))

        dim_0 = 0
        word_sequences = word_sequences[dim_0]
        ngram_sequences = map(get_ngrams_from, word_sequences)
        return ngram_sequences
