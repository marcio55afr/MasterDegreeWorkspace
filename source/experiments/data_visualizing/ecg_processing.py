"""
This is a experiment to proccess a dataset and get
some knowledge about the number of ngram words created
and its importance.

"""
from genericpath import isfile
import sys
sys.path.append('C:\\Users\\marci\\Desktop\\MasterDegreeWorkspace\\source')
import os
import pandas as pd
pd.set_option("display.width", 500)
pd.set_option("max_colwidth", 80)
pd.set_option("max_columns", 10)


from experiments.database.ts_handler import get_dataset
from utils import ResolutionMatrix, ResolutionHandler, NgramExtractor
from transformations import MSAX


def main():

    # handling data
    if(not os.path.isfile('ecg/bag_of_bags_train.csv')):
        train, _ = get_dataset('ECG5000')
        exp_CountUniqueWords(train.data, 'ECG5000 train')
    if(not os.path.isfile('ecg/bag_of_bags_test.csv')):
        _ , test = get_dataset('ECG5000')
        exp_CountUniqueWords(test.data, 'ECG5000 test')

def exp_CountUniqueWords(data, data_name):
    print("\n\nCount unique words experiment\n{} \n\n".format(data_name))
    # initializating useful objects
    rm = ResolutionMatrix(data.iloc[0].size)
    wi, wo =  rm.get_windows_and_words()
    tf = MSAX()
    
    i=0
    n_samples = data.shape[0]
    bobs = pd.DataFrame()
    print("Generating the bag of bags")
    print("__________")
    for ts in data:
        if(i%(n_samples/10) == 0):
            print('#',end='')
        word_seq = tf.transform(ts, wi, wo)
        bob = NgramExtractor.get_bob(word_seq, rm.matrix)
        bob['sample'] = i
        i+=1
        bobs = pd.concat([bobs,bob], axis=0, join='outer', ignore_index=True)
    print("\n Number of unique ngram words in this bag of bags")
    print(bobs['ngram word'].unique().size)
    bobs

if __name__ == "__main__":
    main()