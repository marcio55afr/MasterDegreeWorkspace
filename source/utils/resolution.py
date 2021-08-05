


class ResolutionHandler():

    WINDOW_SPLIT_INDEX=0
    WORD_SPLIT_INDEX=1
    NGRAM_SPLIT_INDEX=2

    @classmethod
    def get_window_from(resolution):
        return int(resolution.split()[ ResolutionHandler.WINDOW_SPLIT_INDEX ])

    @classmethod
    def get_word_from(resolution):
        return int(resolution.split()[ ResolutionHandler.WORD_SPLIT_INDEX ])

    @classmethod
    def get_ngram_from(ngram_resolution):
        return int(ngram_resolution.split()[ ResolutionHandler.NGRAM_SPLIT_INDEX ])