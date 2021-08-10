
# Ngram-Word Frequencies

Here is some results about the frequencies of the ngram words of 3 datasets. These datasets were chosen by the time series length and their type.
The length of the series is the same inside a dataset but vary between them from 140 to 1024. This length was chosen in order to create a bunch of
different words through differents windows to find some representative frequency in there, at the same time it keeps the experiment as fast as possible.
The types were distributed between **_ECG, Sensor and Motion_** aimming data that may have patterns in any ary arbitrary timestamp.
The datasets are **_ECG5000, StarLightCurves and Worms_**, all of them from the well known [UEA & UCR Repository](https://timeseriesclassication.com/).

## First Analysis

The first look at the bag of bags extracted from the datasets, couting the total of words and the frequencies of each word,
comparing the counts of train and test split and what not.

### Total of words

```
ECG5000 - train
>>> train.size
500
>>> get_unique_words(bag_of_bags).size
1367590

ECG5000 - test
>>> train.size
4500
>>> get_unique_words(bag_of_bags).size
11159985
```
