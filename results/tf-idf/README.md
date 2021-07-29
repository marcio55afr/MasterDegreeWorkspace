# TF-IDF

Here is described the conclusions about the indice tf-idf and some results that contributed to it.

## Class Separability

The indice tf-idf calculated for each word consering each class as a document wasn't a good measure to select good words.
These words were extracted from the timeseries using the algorithm SAX with many windows size and also including ngrams words.
The combination of a window and a ngram constitutes a *resolution*.
The tf-idf wasn't neither able to select words or resolutions in order to leverage the accuracy of a model trained with the words selected.
Above is seted the reasons why tf-idf considering class as a document wasn't good:

### Low Frequent Words

Dictionary-based algorithms indentify the frequency of patterns to separate the time series by its class. Given a pattern which appears  
in all objects of many classes, it is possible that a classe can be identified by a low frequency of this pattern relative to the others.
This will give a low tf-idf value difficulting the task of defining a word as good or bad. Since will have good words with low and high tf-idf values.

### Always Frequent Word

Another situation where the tf-idf fails is to calculate the value of a word present in all classes. Due to the formula, the *IDF* will be equal 0.
Therefore all words present in all classes will have the same tf-idf, even when its frequencies is different from each other. 
