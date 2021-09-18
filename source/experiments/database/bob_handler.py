import os
import pandas as pd
import numpy as np
import time

from scipy.sparse import spmatrix, save_npz, load_npz

DATASET_NAMES = ['ECG5000', 'Worms', 'StarLightCurves']
BOB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bob/")
EXTENSION = '.csv'

CONFIG_PATH = '{}/{}/WordLen_{}/'.format
# "dataset_name"/"discretization"/WordLen_6/bob_"split".csv

def write_bag(bag: pd.DataFrame, dataset_name, discretization, wordlen, split):
    
    config_path = CONFIG_PATH(dataset_name, discretization, wordlen)
    folder_path = BOB_PATH+config_path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    #print('Writing the bag of bags...')
    bag_path = folder_path+'bag_{}.csv'.format(split)
    
    bag.to_csv(bag_path)
    
def read_bag(dataset_name, discretization, wordlen, split):
    
    config_path = CONFIG_PATH(dataset_name, discretization, wordlen)
    folder_path = BOB_PATH+config_path
    bag_path = folder_path+'bag_{}.csv'.format(split)
    
    if not os.path.isfile(bag_path):
        raise RuntimeError("Tried to read an inexistent file with path: ",
                           bag_path)
    
    #print('Reading the bag of bags...')
    bag = pd.read_csv(bag_path, index_col = 0)
    return bag
    
    
def write_bob( bob: spmatrix , samples_id, words, dataset_name, discretization, wordlen, split):
    
    config_path = CONFIG_PATH(dataset_name, discretization, wordlen)
    folder_path = BOB_PATH+config_path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    #print('Writing the bag of bags...')
    bob_path = folder_path+'bob_{}.npz'.format(split)
    index_path = folder_path+'indices_{}.csv'.format(split)
    column_path = folder_path+'columns_{}.csv'.format(split)
    
    samples_id = pd.Series(samples_id)
    words = pd.Series(words)
    
    save_npz(bob_path, bob)
    samples_id.to_csv(index_path, index=False)
    words.to_csv(column_path, index=False)

def read_bob(dataset_name, discretization, wordlen, split):
    
    config_path = CONFIG_PATH(dataset_name, discretization, wordlen)
    folder_path = BOB_PATH+config_path
    bob_path = folder_path+'bob_{}.npz'.format(split)
    index_path = folder_path+'indices_{}.csv'.format(split)
    column_path = folder_path+'columns_{}.csv'.format(split)
    
    if not os.path.isfile(bob_path) or not os.path.isfile(index_path) or not os.path.isfile(column_path):
        raise RuntimeError("Tried to read an inexistent file with path: ",
                           bob_path,
                           index_path,
                           column_path)
    
    #print('Reading the bag of bags...')
    bob = load_npz(bob_path)
    indices = pd.read_csv(index_path, index_col = False, squeeze=True)
    columns = pd.read_csv(column_path, index_col = False, squeeze=True)
    return bob, indices.values, columns.values
    
