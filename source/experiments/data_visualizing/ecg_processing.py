"""
This is a experiment to proccess a dataset and get
some knowledge about the number of ngram words created
and its importance.

"""
from genericpath import isfile
import sys
sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace/source')
sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace')
import os
from collections import Counter
import pandas as pd
import numpy as np
pd.set_option("display.width", 500)
pd.set_option("max_colwidth", 80)
pd.set_option("max_columns", 10)


from experiments.database.ts_handler import get_dataset
from utils import ResolutionMatrix, ResolutionHandler, NgramExtractor
from transformations import MSAX


def main():

    DATASETS = ['ECG5000', 'StartLightCurtes', 'Worms']
    for dataset in ['ECG5000']:
        # change the print location
        path='C:/Users/marci/Desktop/MasterDegreeWorkspace/source/experiments/data_visualizing/'
        print_path = path+dataset+'.txt'
        f = open(print_path, 'a')
        sys.stdout = f
        
        print('\n\nDataset: {}'.format(dataset))    
        print('################################')
        train_set,_ = get_dataset(dataset)
        rm = ResolutionMatrix(train_set.iloc[0,0].size)
        del(train_set)
        print('Resolution Matrix')
        print(rm.matrix)
        
        # get the bag of bags
        bob_train, bob_test = _get_bag_of_bags_from(dataset)
        classes = bob_train['label'].unique()
    
        # start the experiments
        exp_CountUniqueWords(bob_train, bob_test)
        exp_CountUniqueWordsByClass(bob_train, bob_test, classes)
        exp_CountUniqueWordsByResolution(bob_train, bob_test)
        exp_CountExclusiveWordByClass(bob_train, bob_test, classes)
        exp_CountAlwaysPresentWordByClass(bob_train, bob_test, classes)
        exp_CountAlmostAlwaysPresentWordByClass(bob_train, bob_test, classes)
        exp_AlwaysPresentWordFrequenciesByClass(bob_train, bob_test, classes)


def exp_CountUniqueWords(bob_train, bob_test):
    
    print("\n\nExperiment - Counting unique words\n")
    
    unique_train = bob_train['ngram word'].unique()
    del(bob_train)
    unique_test = bob_test['ngram word'].unique()
    del(bob_test)

    print('Train unique words: {}'.format(unique_train.size))
    print('Test unique words: {}'.format(unique_test.size))
    intersection = np.intersect1d(unique_train, unique_test)
    print("Intersecting unique words: {}".format(intersection.size))

def exp_CountUniqueWordsByClass(bob_train, bob_test, classes):
    
    print("\n\nExperiment - Counting unique words per class\n")
    
    gp_train = bob_train[['sample','label','ngram']].groupby(['sample','label']).count().groupby('label').count()['ngram']
    gp_test = bob_test[['sample','label','ngram']].groupby(['sample','label']).count().groupby('label').count()['ngram']
    result = pd.DataFrame()
    for c in classes:
        unique_train = bob_train.loc[bob_train['label'] == c,
                                     'ngram word'].unique()
        unique_test = bob_test.loc[bob_test['label'] == c,
                                 'ngram word'].unique()
        result.loc[c,'train'] = unique_train.size
        result.loc[c,'test'] = unique_test.size
        result.loc[c,'intersection'] = np.intersect1d(unique_train,
                                                      unique_test).size
        result.loc[c,'train samples'] = gp_train.loc[c]
        result.loc[c,'test samples'] = gp_test.loc[c]

    result.index.name = 'Class'
    print(result.astype(np.int64))

def exp_CountUniqueWordsByResolution(bob_train, bob_test):
    
    print("\n\nExperiment - Counting unique words per resolution\n")
    
    rm_train = pd.DataFrame(dtype=int)
    rm_test = pd.DataFrame(dtype=int)
    rm_intersection = pd.DataFrame(dtype=int)
    for reso in bob_train.resolution.unique():
        for n in bob_train.loc[bob_train.resolution == reso, 'ngram'].unique():
            # train
            mask = (bob_train.resolution == reso) & (bob_train.ngram == n)
            unique_words_train = bob_train.loc[mask, 'ngram word'].unique()
            rm_train.loc[n,reso] = unique_words_train.size
            # test
            mask = (bob_test.resolution == reso) & (bob_test.ngram == n)
            unique_words_test = bob_test.loc[mask, 'ngram word'].unique()
            rm_test.loc[n,reso] = unique_words_test.size
            # intersection
            unique_words = np.intersect1d(unique_words_train,
                                          unique_words_test).size
            rm_intersection.loc[n,reso] = unique_words
    
    rm_train.fillna(0).astype(np.int64).columns.name = 'train'
    rm_test.fillna(0).astype(np.int64).columns.name = 'test'
    rm_intersection.fillna(0).astype(np.int64).columns.name = 'intersection'
    print(rm_train,end='\n\n')
    print(rm_test,end='\n\n')
    print(rm_intersection,end='\n\n')
    
def exp_CountAlwaysPresentWordByClass(bob_train, bob_test, classes):    
    print("\n\nExperiment - Counting words always present in each class\n")
    
    def _get_always_present_word(data, c):
        equaL_class_sample = data.loc[data['label'] == c]
        n_samples = equaL_class_sample['sample'].unique().size
        class_words = Counter(equaL_class_sample['ngram word'])
        class_words = pd.Series(class_words)
        ap_word = class_words.loc[class_words == n_samples]
        return ap_word
    
    result = pd.DataFrame(dtype=np.int64)
    for c in classes:
        ap_word_train = _get_always_present_word(bob_train, c)
        ap_word_test = _get_always_present_word(bob_test, c)
        
        result.loc[c,'Train'] = ap_word_train.size
        result.loc[c,'Test'] = ap_word_test.size
        result.loc[c,'Intersection']  = np.intersect1d(ap_word_train.index,
                                                       ap_word_test.index).size 
    result.index.name = 'Class'
    print(result.astype(np.int64))
    
def exp_CountExclusiveWordByClass(bob_train, bob_test, classes):    
    print("\n\nExperiment - Counting exclusive words of each class\n")
    
    def _get_exclusive_word(data, c):
        mask = data['label'] == c
        equaL_class_sample = data.loc[ mask ]
        others_classes_sample = data.loc[ ~mask ]
        
        class_words =  Counter(equaL_class_sample['ngram word'])
        other_classes_words = Counter(others_classes_sample['ngram word'])
        
        ex_words = class_words - (class_words&other_classes_words)
        return pd.Series(ex_words)
    
    result = pd.DataFrame(dtype=np.int64)
    for c in classes:
        ex_word_train = _get_exclusive_word(bob_train, c)
        ex_word_test = _get_exclusive_word(bob_test, c)
        
        result.loc['Train',c] = ex_word_train.size
        result.loc['Test',c] = ex_word_test.size
        result.loc['Intersection',c] = np.intersect1d(ex_word_train.index,
                                             ex_word_test.index).size
    result.index.name = 'Class'
    print(result.astype(np.int64))
    
def exp_CountAlmostAlwaysPresentWordByClass(bob_train, bob_test, classes):    
    print("\n\nExperiment - Counting words almost always present in each class\n")
    
    def _get_almost_always_present_word(data, c):
        equaL_class_sample = data.loc[data['label'] == c]
        n_samples = equaL_class_sample['sample'].unique().size
        class_words = Counter(equaL_class_sample['ngram word'])
        class_words = pd.Series(class_words)
        ap_word = class_words.loc[class_words >= .8*n_samples]
        return ap_word
    
    result = pd.DataFrame(dtype=np.int64)
    for c in classes:
        ap_word_train = _get_almost_always_present_word(bob_train, c)
        ap_word_test = _get_almost_always_present_word(bob_test, c)
        intersecting_words = np.intersect1d(ap_word_train.index,
                                            ap_word_test.index)
        result.loc['Train',c] = ap_word_train.size
        result.loc['Test',c] = ap_word_test.size
        result.loc['Intersection',c] = len(intersecting_words)
    result.index.name = 'Class'
    print(result.astype(np.int64))

def exp_AlwaysPresentWordFrequenciesByClass(bob_train, bob_test, classes):    
    print("\n\nExperiment - Counting the frequency of words always present in each class\n")
    
    def _get_relative_frequency_of_always_present_word(data, c):
        equaL_class_sample = data.loc[data['label'] == c]
        n_samples = equaL_class_sample['sample'].unique().size
        class_words = Counter(equaL_class_sample['ngram word'])
        class_words = pd.Series(class_words)
        ap_word = class_words.loc[class_words >= n_samples]
        
        equaL_class_words = equaL_class_sample.set_index('ngram word')
        equaL_class_words = equaL_class_words.loc[ap_word.index,'frequency']
        word_frequencies = equaL_class_words.groupby('ngram word').sum()
        return word_frequencies, n_samples
    
    for c in classes:
        ap_word_freq_train, train_samples = _get_relative_frequency_of_always_present_word(bob_train, c)
        ap_word_freq_test, test_samples = _get_relative_frequency_of_always_present_word(bob_test, c)
        intersecting_words = np.intersect1d(ap_word_freq_train.index,
                                            ap_word_freq_test.index)
        print('\n\nClass :{}\n'.format(c))
        
        ap_word_freq_train.name = 'Train'
        ap_word_mean_freq = ap_word_freq_train/train_samples
        ap_word_mean_freq.name = 'Mean'
        train = pd.concat([ap_word_freq_train,ap_word_mean_freq],axis=1)
        
        ap_word_freq_test.name = 'Test'
        ap_word_mean_freq = ap_word_freq_test/test_samples
        ap_word_mean_freq.name = 'Mean'
        test = pd.concat([ap_word_freq_train,ap_word_mean_freq],axis=1)
        
        result = pd.concat([train, test],axis=1)
        result = result.fillna(0).astype(np.int64)
        result.loc[ intersecting_words, 'Intersection' ] = intersecting_words
        result['train_samples'] = int(train_samples)
        result['test_samples'] = int(test_samples)
        result = result.fillna('')
        print(result)

def _get_labels_from(dataset):
    train_set, test_set = get_dataset(dataset)
    return train_set.target, test_set.target

def _get_bag_of_bags_from(dataset):
    path = 'C:/Users/marci/Desktop/MasterDegreeWorkspace/source/experiments/data_visualizing/'
    folder = {
        'ECG5000' : path+'ecg',
        'Worms': path+'worms'
        }
    
    bob_train_path = folder[dataset]+'/bag_of_bags_train.csv'
    bob_test_path = folder[dataset]+'/bag_of_bags_test.csv'
    
    if(os.path.isfile(bob_train_path) and os.path.isfile(bob_test_path)):
        bob_train = pd.read_csv(bob_train_path)
        bob_test = pd.read_csv(bob_test_path)
        return bob_train, bob_test
    
    train_set, test_set = get_dataset(dataset)
    train_labels, test_labels = _get_labels_from(dataset)
    
    bob_train = _extract_bob_from(train_set.data)
    bob_train['label'] = train_labels.loc[bob_train['sample']].values
    print('\nWriting down the bag of bags of the train part')
    bob_train.to_csv(bob_train_path, index=False)

    bob_test = _extract_bob_from(test_set.data)
    bob_test['label'] = test_labels.loc[bob_test['sample']].values
    print('\nWriting down the bag of bags of the test part')
    bob_test.to_csv(bob_test_path, index=False)
    return bob_train, bob_test

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