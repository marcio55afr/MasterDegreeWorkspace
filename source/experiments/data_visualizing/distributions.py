"""


"""

import sys

import os
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from source.experiments.database.ts_handler import get_dataset
from source.utils import ResolutionMatrix, ResolutionHandler, NgramExtractor
from source.transformations import MSAX


def main():
    path = 'C:/Users/marci/Desktop/MasterDegreeWorkspace/source/experiments/data_visualizing/'
    folder = {
        'ECG5000': path + 'ecg/',
        'Worms': path + 'worms/'
    }
    DATASETS = ['ECG5000', 'StartLightCurtes', 'Worms']

    for dataset in ['ECG5000']:
        folder_path = folder[dataset]
        bob_train = _get_bag_of_bags_from(dataset, folder_path)
        classes = bob_train.label.unique()

        # start the experiments
        # exp_HightPresentWordDistribution(bob_train, classes)
        exp_wordpresence(bob_train, classes)


def exp_HightPresentWordDistribution(bob_train, classes):
    high_freq = ['21 aabcdd', '36 aabcdd', '21 ddcbaa', '6 ddcbaa']
    medium_freq = ['21 ddcbaa', '36 aabcdd', '51 aabcdd', '51 abbcdd', '81 aabcdd', '6 aabcdd']

    for word in high_freq:
        for c in classes:
            data_by_class = bob_train.loc[bob_train['label'] == c]
            word_sample = data_by_class.loc[data_by_class['ngram word'] == word]
            n_samples = word_sample.shape[0]
            word_group = word_sample.groupby('frequency')
            relative_counting = word_group.count()['sample'] / n_samples
            relative_counting.plot()
        plt.show()

    for word in medium_freq:
        for c in classes:
            class_sample = bob_train.loc[bob_train['label'] == c]
            word_group = class_sample.loc[class_sample['ngram word'] == word].groupby('frequency')
            word_group.count()['sample'].plot()
        plt.show()


def exp_wordpresence(bob_train, classes):
    bob_train


def _get_bag_of_bags_from(dataset, folder_path):
    bob_train_path = folder_path + '/bag_of_bags_train.csv'

    if os.path.isfile(bob_train_path):
        bob_train = pd.read_csv(bob_train_path)
        return bob_train

    raise RuntimeError('Bag of bags not found in the path: {}'.format(bob_train_path))


if __name__ == "__main__":
    main()
