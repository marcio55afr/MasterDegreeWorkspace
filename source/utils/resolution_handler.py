import numpy as np
import pandas as pd

class ResolutionHandler():

    # Class attributes

    WINDOW_SPLIT_INDEX=0
    NGRAM_SPLIT_INDEX=1  
    
    
    # Class functions
    
    def get_ngrams_from(matrix, window):
        mask = matrix[window] > False
        return matrix.loc[mask, window].index.values
    
    def get_resolution_from(word):
        s = word.split()
        return '{} {}'.format( s[0], s[1] )
    
    def get_window_from(resolution):
        """
        Parameters
        ----------
        resolution : str
            Resolution as string with one window and word lengths.

        Returns
        -------
        window : int
            Return only the window length.

        """
        window = int(resolution.split()[ ResolutionHandler.WINDOW_SPLIT_INDEX ])
        return window

    def get_ww_from(resolution):
        """
        ww - abreviation of Window Word

        Parameters
        ----------
        resolution : str
            String with window and word lengths, in this order, splited by a
            space

        Returns
        -------
        window : int
            window length.
        word : TYPE
            word length.

        """
        split = resolution.split()
        window = int(split[ ResolutionHandler.WINDOW_SPLIT_INDEX ])
        word = int(split[ ResolutionHandler.WORD_SPLIT_INDEX ])
        return window, word
    
    def get_wwn_from(ngram_resolution):
        """
        wwn - abreviation of Window Word Ngram
        
        Parameters
        ----------
        ngram_resolution : str
            String with the lengths of window word and ngram in this order.
            Splited by a space

        Returns
        -------
        tuple : int, int, int
            Return the window, word and ngram
            
        """
        split = ngram_resolution.split()
        return (int(split[ ResolutionHandler.WINDOW_SPLIT_INDEX ]),
                int(split[ ResolutionHandler.WORD_SPLIT_INDEX ]),
                int(split[ ResolutionHandler.NGRAM_SPLIT_INDEX ]))
    

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

    def generate_resolution_matrix(window_lens, max_ngram) -> pd.DataFrame:
        
        # Defining the indexes bases on the ngrams used
        idx = range(1, max_ngram + 1)
        
        # Creating the dataframe setting the possibles ngram resolutions as 1
        # and the rest as False
        biggest_window = window_lens[-1]
        matrix = pd.DataFrame(False, index=idx, columns=window_lens)
        for j in range(matrix.shape[1]):
            max_ngram = biggest_window//(window_lens[j])
            matrix.iloc[0:max_ngram, j] = 1
        
        return matrix