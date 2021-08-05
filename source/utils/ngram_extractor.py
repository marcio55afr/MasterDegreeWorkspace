
import numpy as np
import pandas as pd

from utils.resolution import ResolutionHandler

class NgramExtractor(object):


    def get_bonw(word_sequence, reso_matrix):
        '''
        This function receives a multiresolution discretization of a
        single time series and a resolution matrix of ngram-resolutions.
        It will return the valid ngram-resolutions pointed in the matrix
        of each resolution inside the multiresolution discretization.

        Parameters
        ----------
            word_sequence : dict
                The key is the resolution and its content is a word
                sequence discretized using this key.
            reso_matrix : pandas.DataFrame
                Columns are resolutions and indexes are ngrams. Each cell
                must be a boolean validating the correspondent
                ngram-resolution
            
        Returns
        -------
            bag_of_bags : DataFrame
                Return a bag of bags as a DataFrame containing all ngram-word
                frequencies of each resolution to a specific word sequence. 

        '''

        if(type(word_sequence) != dict):
            raise TypeError('The parameter word_sequence needs to be a '+
            'multiresolution discretization as a dict format.')
        
        if(type(reso_matrix) != pd.DataFrame):
            raise TypeError('The parameter reso_matrix needs to be a '+
            'DataFrame')

        expec_reso = reso_matrix.columns
        recev_reso = word_sequence.keys()

        for resolution in recev_reso:
            if(resolution not in expec_reso):
                raise RuntimeError('Received a resolution within the '+
                'word_sequence that is not expected in the reso_matrix.')
        
        
        # Loop for processing each sequence related to each resolution
        # one by one.
        bag_of_bags = pd.DataFrame()
        for resolution in recev_reso:

            # variables
            sequence = word_sequence[resolution]
            window_len = ResolutionHandler.get_window_from(resolution)
            valid_ngrams = reso_matrix[resolution]
            
            # create and count all valid ngrams for this sequence
            bag_of_ngrams = NgramExtractor.get_ngram_frequency(sequence,
                                                               window_len,
                                                               valid_ngrams)
            bag_of_ngrams['resolution'] = resolution

            # concatenate all bag of ngram words in the same dataframe
            bag_of_bags = bag_of_bags.append(bag_of_ngrams, ignore_index=True)

        return bag_of_bags


    def get_ngram_frequency(sequence: pd.Series, window_len, valid_ngrams):

        if(type(sequence)!=pd.Series):
            raise TypeError('The sequence of words must be a pandas Series')

        # variables
        seq_len = len(sequence)
        bag_of_ngrams = pd.DataFrame()
        
        # loop to process each ngram at a time
        for n in valid_ngrams:
            # It is necessary to verify this?
            # check 
            if((seq_len -(n-1)*window_len) <= 0):
                break

            # Create and count all word with the specific n-gram
            ngram_freq = dict()
            for j in range(seq_len -(n-1)*window_len):
                ngram = ' '.join(sequence.iloc[np.asarray(range(n))*window_len + j])
                ngram_freq[ngram] = ngram_freq.get(ngram,0) + 1
                # Second Paper - technique ability
                # todo - assign on the feature its dimension id

            # Set the bag as a DataFrame and add some informations
            df = pd.DataFrame.from_dict(ngram_freq, orient='index', columns=['frequency'])
            df.index.name = 'ngram word'
            df = df.reset_index()
            df['ngram'] = n

            # Concatenate all dataframes
            bag_of_ngrams = bag_of_ngrams.append(df, ignore_index=True)

        return bag_of_ngrams
