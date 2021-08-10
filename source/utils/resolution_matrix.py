
import math
from utils.resolution_handler import ResolutionHandler

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
        self.biggest_window = math.floor(ts_length*biggest_window_prop)
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

        window_lengths = ResolutionHandler.generate_window_lengths(self.smallest_window,
                                                           self.biggest_window,
                                                           self.num_windows)
        word_lengths = ResolutionHandler.generate_word_lengths(window_lengths,
                                                       self.smallest_word,
                                                       self.dimension_reduction)
        matrix = ResolutionHandler.generate_resolution_matrix(window_lengths,
                                                              word_lengths,
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