import time
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype

class NgramExtractor(object):


    def get_bob(word_sequences, resolution_matrix, verbose=True):
        '''
        bob refers to bag of bags.
        This function receives a multiresolution discretization of a
        single time series and a ngram-multiresolution matrix. It will
        return the ngram-word frequencies for all valid ngram-resolution
        that consist in the resolution present in the discretization 
        word_sequence and a possible ngram present in the reso_matrix.
        The ngram-resolution is used as a coordinate in the
        ngram-resolution matrix and it is valid if the ceil is True.

        Parameters
        ----------
            word_sequence : dict
                The key is the resolution and its content is a word
                sequence discretized using this key.
            resolution_matrix : pandas.DataFrame
                Columns are resolutions and indexes are ngrams. Each cell
                must be a boolean validating the correspondent
                ngram-resolution
            
        Returns
        -------
            bag_of_bags : DataFrame
                Return a bag of bags as a DataFrame containing all ngram-word
                frequencies of each resolution to a specific word sequence. 

        '''

        #if(type(word_sequences) != dict):
        #    raise TypeError('The parameter word_sequence needs to be a '+
        #    'multiresolution discretization as a dict format.')
        
        if(type(resolution_matrix) != pd.DataFrame):
            raise TypeError('The parameter reso_matrix needs to be a '+
            'DataFrame')

        windows_expected = resolution_matrix.columns
        windows_received = word_sequences.window.unique()

        for window in windows_received:
            if(window not in windows_expected):
                raise RuntimeError('A sequence was transformed using a '+
                'window length unexpected by the reso_matrix passed as a parameter')
        
        # TODO expand to multivariate ngram_extractor
        dim=0
        if verbose:
            print('Extracting the ngrams...')
            print('__________', end='')
            print('')
        samples = word_sequences.index.unique()
        n = samples.size
        v=1
        # Loop for processing all sequences of each sample
        bag_of_bags = pd.DataFrame()
        for sample in samples:
            sample_sequences = word_sequences.loc[sample]
            
            bag_of_ngrams = []
            for i in range(sample_sequences.shape[0]):
                # variables
                row = sample_sequences.iloc[i]
                sequence = row.loc[dim]
                window = row.loc['window']
                valid_mask = resolution_matrix[window] == 1
                ngrams = resolution_matrix[valid_mask].index
            
                #TODO test sequential ngrams
                #     test ngrams with space of the window size
                #     test ngrams between different windows sizes
                # create and count all valid ngrams for this sequence
                bonw = NgramExtractor.get_bonw(sequence,
                                               window,
                                               ngrams)
                bonw['window'] = window
                
                # concatenate all bag of ngram words in the same dataframe
                bag_of_ngrams.append(bonw)
            bag_of_ngrams = pd.concat(bag_of_ngrams,
                                      ignore_index=False,
                                      axis=0)
            bag_of_ngrams['sample'] = sample
            bag_of_bags = pd.concat([bag_of_bags, bag_of_ngrams],
                                    axis=0,
                                    ignore_index=False)
            v+=1
            if verbose:
                if(v%(n//10) == 0):
                    print('#', end='')
        if verbose: print('')
        
        #return bag_of_bags
        bag_of_bags = bag_of_bags.reset_index()
        print('Creating the feature sparse matrix...')
        
        samples_ordered = sorted(bag_of_bags['sample'].unique())
        ngram_words_ordered = sorted(bag_of_bags['index'].unique())
        
        sample_c = CategoricalDtype( samples_ordered, ordered=True)
        ngram_word_c = CategoricalDtype( ngram_words_ordered, ordered=True)
        
        row = bag_of_bags['sample'].astype(sample_c).cat.codes
        col = bag_of_bags['index'].astype(ngram_word_c).cat.codes
        sparse_matrix = csr_matrix((bag_of_bags['frequency'],(row, col)),
                                   shape=(sample_c.categories.size,
                                          ngram_word_c.categories.size)
                                   )

        return sparse_matrix, samples_ordered, ngram_words_ordered


    def get_bonw(sequence, window_len, ngrams):
        '''
        bonw refers to bag of ngram words.
        This function receives a sequence of words discretized with only
        one resolution and based on the window length used in the
        discretization it will create the ngrams. The ngrams are created
        without any intersection of windows inside itself but there is
        interesections between different ngrams. The 'n' used is passed
        as a list, valid_ngrams, and the occourence of each one is counted
        using a dictionary and then transformed to a DataFrame in order
        to save some info. 

        Parameters
        ----------
            sequence : pandas.Series
                Word sequence discretized in a single resolution.                
            window_len : pandas.DataFrame
                Length of the window used to discretize the sequence.
                Used to avoid intersection of windows inside a ngram.
            valid_ngrams : list or iterator of int
                Set of ngrams to create the ngram words.
            
        Returns
        -------
            bag_of_ngram_words : DataFrame
                Return a bonw as a DataFrame containing all ngram-word
                frequencies of a word sequence in a specific resolution. 

        '''

        #if(type(sequence)!=pd.Series):
        #    raise TypeError('The sequence of words must be a pandas Series')
        if( (ngrams < 1).any() ):
            raise ValueError('All ngrams must be greater than or equal to 1')

        # variables
        seq_len = len(sequence)
        bag_of_ngram_words = pd.DataFrame()
        # loop to process each ngram at a time
        for n in ngrams:
            # It is necessary to verify this?
            # check 
            if((seq_len -(n-1)*window_len) <= 0):
                break

            # Create and count all word with the specific n-gram
            nw_freq = dict()
            for j in range(seq_len -(n-1)*window_len):
                ngram_word = ' '.join(sequence[np.arange(n)*window_len + j])
                nw_freq[ngram_word] = nw_freq.get(ngram_word,0) + 1
                # Second Paper - technique ability
                # todo - assign on the feature its dimension id

            # Set the bag as a DataFrame and add some informations
            df = pd.DataFrame.from_dict(nw_freq, orient='index', columns=['frequency'])
            df['ngram'] = n
            
            # Concatenate all dataframes
            bag_of_ngram_words = pd.concat([bag_of_ngram_words, df],
                                           ignore_index=False,
                                           axis=0)
        return bag_of_ngram_words
