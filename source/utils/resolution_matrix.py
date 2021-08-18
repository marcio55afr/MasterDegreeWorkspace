
import math
from utils.resolution_handler import ResolutionHandler

class ResolutionMatrix(object):
        
    def __init__(self,
                 ts_length,
                 word_size = 8,
                 num_windows = 10,
                 max_window_length = None
                 ):
        self.word_size = word_size
        self.num_windows = 10 # Fixed number of windows
        self.smallest_window = word_size
        if max_window_length is None:
            self.biggest_window = ts_length
        else:
            self.biggest_window = max_window_length
        self.max_ngram = self.biggest_window//self.smallest_window
        self.matrix = self.create_matrix(ts_length)

    def create_matrix(self, ts_length):        
        if (self.biggest_window-self.smallest_window) < self.num_windows:
            raise ValueError('the difference between the smallest windows {}'.format(self.smallest_window)+
            'and the greatest window {} must be greater than {}'.format(self.biggest_window,
                                                                        self.num_windows))

        window_lengths = ResolutionHandler.generate_window_lengths(self.smallest_window,
                                                           self.biggest_window,
                                                           self.num_windows)
        #word_lengths = ResolutionHandler.generate_word_lengths(window_lengths,
        #                                               self.smallest_word,
        #                                               self.dimension_reduction)
        matrix = ResolutionHandler.generate_resolution_matrix(window_lengths,
                                                         self.max_ngram)
        return matrix
    
    def get_windows_and_words(self):
        windows = []
        words = []
        columns = self.matrix.columns
        for resolution in columns:
            wi,wo = ResolutionHandler.get_ww_from(resolution)
            windows.append(wi)
            words.append(wo)
        
        return windows, words