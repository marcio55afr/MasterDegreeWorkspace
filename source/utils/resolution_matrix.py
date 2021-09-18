
from source.utils import ResolutionHandler

class ResolutionMatrix(object):
        
    def __init__(self,
                 ts_length,
                 word_length,
                 num_windows = None,
                 max_window_length = None
                 ):
        self.word_length = word_length
        self.smallest_window = word_length
        
        if max_window_length is None:
            self.biggest_window = ts_length//2
        else:
            self.biggest_window = max_window_length
            
        if num_windows is not None:
            self.num_windows =  num_windows 
        else:
            self.num_windows = (self.biggest_window-self.smallest_window )//2
        self.num_windows = min(20, self.num_windows)
            
        self.max_ngram = min(5, self.biggest_window//self.smallest_window)
        self.matrix = self.create_matrix(ts_length)

    def create_matrix(self, ts_length):        
        if (self.biggest_window-self.smallest_window) < self.num_windows:
            raise ValueError(
                f'the difference between the smallest window {self.smallest_window} '
                f'and the greatest window {self.biggest_window} must be greater than {self.num_windows}'
            )

        window_lengths = ResolutionHandler.generate_window_lengths(self.smallest_window,
                                                           self.biggest_window,
                                                           self.num_windows)
        #word_lengths = ResolutionHandler.generate_word_lengths(window_lengths,
        #                                               self.smallest_word,
        #                                               self.dimension_reduction)
        matrix = ResolutionHandler.generate_resolution_matrix(window_lengths,
                                                         self.max_ngram)
        return matrix
        
    def get_windows(self):
        
        num_resolutions = self.matrix.sum()
        used_columns = num_resolutions[num_resolutions > 0].index.values
        return used_columns
    
    def get_windows_and_words(self):
        windows = []
        words = []
        columns = self.matrix.columns
        for resolution in columns:
            wi,wo = ResolutionHandler.get_ww_from(resolution)
            windows.append(wi)
            words.append(wo)
        
        return windows, words