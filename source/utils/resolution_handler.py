
import math
import numpy as np

class ResolutionHandler():

    # Class attributes

    WINDOW_SPLIT_INDEX=0
    WORD_SPLIT_INDEX=1
    NGRAM_SPLIT_INDEX=2

    # Class functions

    def create_matrix(ts_length, dimension_reduction=.8):
        min_window_len = 4
        max_window_len = ts_length//2
        num_windows = 10
        min_word_len = 2
        min_ngram = 1
        max_ngram = round(max_window_len/min_window_len)
        if (max_window_len-min_window_len) < 10:
            raise ValueError('the difference between the smallest windows {}'+
            'and the greatest window {} must be greater than {}'.format(min_window_len,
                                                                        max_window_len,
                                                                        num_windows))

        window_lens = ResolutionHandler._get_window_lengths(min_window_len, max_window_len, num_windows)
        word_lens = ResolutionHandler._get_word_lengths(window_lens, min_word_len, dimension_reduction)

        matrix = ResolutionHandler._get_dataframe(window_lens, word_lens, min_ngram, max_ngram)

    def get_window_from(resolution):
        return int(resolution.split()[ ResolutionHandler.WINDOW_SPLIT_INDEX ])

    def get_word_from(resolution):
        return int(resolution.split()[ ResolutionHandler.WORD_SPLIT_INDEX ])

    def get_ngram_from(ngram_resolution):
        return int(ngram_resolution.split()[ ResolutionHandler.NGRAM_SPLIT_INDEX ])    

    def _get_window_lengths(min_window_len, max_window_len, num_windows):       
        windows = []
        window_len = min_window_len
        for i in range(num_windows):
            windows.append(window_len)
            stepwise = (max_window_len-min_window_len)/num_windows
            stepwise = min(1,round(stepwise))
            window_len += stepwise
        return windows
    
    def _get_word_lengths(window_lengths, min_word_length, dimension_reduction):
        words = []
        word_len = min(min_word_length, round(window_lengths[0]*dimension_reduction))
        for w in window_lengths[1:]:
            words.append(word_len)
            stepwise = min(1, round(w*dimension_reduction)-word_len)
            word_len += stepwise
        return words
    
    def _get_dataframe(window_lens, word_lens, min_ngram, max_ngram):
        if(len(window_lens)!=len(word_lens)):
            raise RuntimeError('The quantity of window and word lengths should be the same')
        
        cols = []
        for i in range(len(window_lens)):
            cols.append( str(window_lens[i]) + str(word_lens[i]) )