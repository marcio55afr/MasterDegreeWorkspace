
import math
import numpy as np
import pandas as pd

class ResolutionMatrix(object):
        
    def __init__(self,
                 ts_length,
                 smallest_window = 4,
                 biggest_window_prop = 1/2,
                 max_num_windows = 10,
                 smallest_word = 2,
                 dimension_reduction = .8
                 ):
        self.smallest_window = smallest_window
        self.biggest_window_prop = biggest_window_prop
        self.biggest_window = ts_length*biggest_window_prop
        self.num_windows = 10 # Fixed number of windows
        self.smallest_word = smallest_word
        self.dimension_reduction = dimension_reduction
        
        self.max_ngram = self.biggest_window//self.smallest_window
        self.matrix = self.create_matrix(ts_length)

    def create_matrix(self, ts_length):        
        if (self.biggest_window-self.smallest_window) < self.num_windows:
            raise ValueError('the difference between the smallest windows {}'.format(self.smallest_window)+
            'and the greatest window {} must be greater than {}'.format(self.biggest_window,
                                                                        self.num_windows))

        window_lengths = ResolutionHandler.get_window_lengths(self.smallest_window,
                                                           self.biggest_window,
                                                           self.num_windows)
        word_lengths = ResolutionHandler.get_word_lengths(window_lengths,
                                                       self.smallest_word,
                                                       self.dimension_reduction)
        matrix = ResolutionHandler.generate_resolution_matrix(window_lengths,
                                                              word_lengths,
                                                              self.max_ngram)
        return matrix

class ResolutionHandler():

    # Class attributes

    WINDOW_SPLIT_INDEX=0
    WORD_SPLIT_INDEX=1
    NGRAM_SPLIT_INDEX=2  
    
    
    # Class functions

    def get_window_from(resolution):
        return int(resolution.split()[ ResolutionHandler.WINDOW_SPLIT_INDEX ])

    def get_word_from(resolution):
        return int(resolution.split()[ ResolutionHandler.WORD_SPLIT_INDEX ])

    def get_ngram_from(ngram_resolution):
        return int(ngram_resolution.split()[ ResolutionHandler.NGRAM_SPLIT_INDEX ])    

    def generate_window_lengths(min_window_len, max_window_len, num_windows):       
        windows = []
        window_len = min_window_len
        for i in range(num_windows-1):
            windows.append(window_len)
            stepwise = (max_window_len-window_len)/(num_windows-i-1)
            stepwise = max(1,round(stepwise))
            window_len += stepwise
        windows.append(max_window_len)
        return windows
    
    def generate_word_lengths(window_lengths, min_word_length, dimension_reduction):
        words = []
        word_len = max(min_word_length, round(window_lengths[0]*dimension_reduction))
        for w in window_lengths[1:]:
            words.append(word_len)
            stepwise = max(1, round(w*dimension_reduction)-word_len)
            word_len += stepwise
        words.append(word_len)
        return words
    
    def generate_resolution_matrix(window_lens, word_lens, max_ngram) -> pd.DataFrame:
        if(len(window_lens)!=len(word_lens)):
            raise RuntimeError('The quantity of window and word lengths should be the same')
        
        # Defining the column names based on the window and word size
        cols = []
        for i in range(len(window_lens)):
            window_len = window_lens[i]
            word_len = word_lens[i]
            if( word_len>window_len ):
                raise RuntimeError('The word is greater than the window')
            cols.append( '{} {}'.format(window_len, word_len) )
        
        # Defining the indexes bases on the ngrams used
        idx = np.arange(max_ngram) + 1
        
        # Creating the dataframe setting the possibles ngram resolutions as 1
        # and the rest as False
        biggest_window = window_lens[-1]
        matrix = pd.DataFrame(False, index=idx, columns=cols)
        for j in range(matrix.shape[1]):
            max_ngram = biggest_window//(window_lens[j])
            matrix.iloc[0:max_ngram, j] = 1
        
        return matrix