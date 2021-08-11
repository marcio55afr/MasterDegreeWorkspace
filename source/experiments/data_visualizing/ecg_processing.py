"""
This is a experiment to proccess a dataset and get
some knowledge about the number of ngram words created
and its importance.

"""
from genericpath import isfile
import sys
sys.path.append('C:\\Users\\marci\\Desktop\\MasterDegreeWorkspace\\source')
import os
import numpy
import pandas as pd
pd.set_option("display.width", 500)
pd.set_option("max_colwidth", 80)
pd.set_option("max_columns", 10)


from experiments.database.ts_handler import get_dataset
from utils import ResolutionMatrix, ResolutionHandler, NgramExtractor
from transformations import MSAX


def main():

    DATASETS = ['ECG5000', 'StartLightCurtes', 'Worms']
    
    dataset = 'ECG5000'
    #exp_CountUniqueWords(dataset)
    #exp_CountIntersectingWords(dataset)
    exp_CountUniqueWordsByClass(dataset)
    #exp_CountIntersectingWordsByClass(dataset)

def exp_CountUniqueWords(dataset):
    
    print("\n\nCount unique words experiment\n")
    bob_train, bob_test = _get_bag_of_bags_from(dataset)
    print("{} - train".format(dataset))
    print("Unique ngram words: {}".format(bob_train['ngram word'].unique().size))
    print("{} - test".format(dataset))
    print("Unique ngram words: {}".format(bob_test['ngram word'].unique().size))

def exp_CountIntersectingWords(dataset):
    
    print("\n\nExperiment - counting intersecting unique words\n")
    bob_train, bob_test = _get_bag_of_bags_from(dataset)
    
    unique_train = bob_train['ngram word'].unique()
    del(bob_train)
    unique_test = bob_test['ngram word'].unique()
    del(bob_test)
    
    print("{}".format(dataset))
    intersection = numpy.intersect1d(unique_train, unique_test)
    print("Unique ngram words: {}".format(intersection.size))

def exp_CountUniqueWordsByClass(dataset):
    
    print("\n\nExperiment - Counting unique words per class\n")
    # get data
    bob_train, bob_test = _get_bag_of_bags_from(dataset)
    train_labels, test_labels = _get_labels_from(dataset)
    train_classes = train_labels.unique()
    test_classes = test_labels.unique()
    
    bob_train['label'] = train_labels.loc[bob_train['sample']].values
    bob_test['label'] = test_labels.loc[bob_test['sample']].values
    
    count0 = pd.Series()
    count1 = pd.Series()
    for c in train_classes:
        count0.loc[c] = bob_train.loc[bob_train['label'] == c,
                                  'ngram word'].unique().size
        count1[c] = bob_test.loc[bob_test['label'] == c,
                                 'ngram word'].unique().size
    
    result = pd.DataFrame()
    result['train'] = count0
    result['test'] = count1
    
    print(result)  

def _get_labels_from(dataset):
    train_set, test_set = get_dataset(dataset)
    return train_set.target, test_set.target

def _get_bag_of_bags_from(dataset):    
    folder = {
        'ECG5000' : 'ecg'
        }
    
    bob_train_path = folder[dataset]+'/bag_of_bags_train.csv'
    bob_test_path = folder[dataset]+'/bag_of_bags_test.csv'
    
    if(os.path.isfile(bob_train_path) and os.path.isfile(bob_test_path)):
        bob_train = pd.read_csv(bob_train_path)
        bob_test = pd.read_csv(bob_test_path)
        return bob_train, bob_test
    
    train_set, test_set = get_dataset(dataset)
    bob_train = _extract_bob_from(test_set.data)
    bob_test = _extract_bob_from(train_set.data)
    print('\nWriting down the bag of bags of the train part')
    bob_train.to_csv(bob_train_path)
    print('\nWriting down the bag of bags of the test part')
    bob_test.to_csv(bob_test_path)

def _extract_bob_from(timeseries_set):
    rm = ResolutionMatrix(timeseries_set.iloc[0].size)
    wi, wo =  rm.get_windows_and_words()
    tf = MSAX()
    
    i=0
    n_samples = timeseries_set.shape[0]
    bag_of_bags = pd.DataFrame()
    print("Generating the bag of bags")
    print("__________")
    for ts in timeseries_set:
        if(i%(n_samples/10) == 0):
            print('#',end='')
        word_seq = tf.transform(ts, wi, wo)
        bob = NgramExtractor.get_bob(word_seq, rm.matrix)
        bob['sample'] = i
        i+=1
        bag_of_bags = pd.concat([bag_of_bags,bob], axis=0, join='outer', ignore_index=True)
    
    return bag_of_bags
    
    

if __name__ == "__main__":
    main()