import os
from sktime.benchmarking.data import UEADataset
import pandas as pd
import numpy as np

DATASET_NAMES = ['Worms', 'StarLightCurves', 'ECG5000']
TS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Univariate_ts\\")
DATASETS = [UEADataset(path=TS_PATH, name=name) for name in DATASET_NAMES]

HDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hdf\\")
EXTENSION = '.h5'
KEYS = ['train', 'test']

def get_dataset( dataset_name ) -> tuple:
    path = HDF_PATH + dataset_name + EXTENSION
    train = pd.read_hdf(path,key='train')
    test = pd.read_hdf(path,key='test')
    return train, test

def write_ts( data: pd.DataFrame, path, key):
    data.index = range(data.shape[0])
    data.columns = ['data','target']
    data = data.astype({'target': np.int32})
    data.to_hdf(path, key=key)

def get_dataframe_from_tsfiles():
    """
    Execute this functions with the datasets as dataset.ts inside a folder
    specified in the variable TS_PATH.
    Then will be created a folder if not exists and there is saved the dataset
    as a hdf5 file separeted by train and test using the key parameter.
    
    Warnings
    --------
    The PyTables cannot recognize a column of pandas.Series, as it is.
    So it rises a warning about performance that I cannot solve yet.

    Returns
    -------
    None.

    """
    
    if not os.path.exists(HDF_PATH):
        os.mkdir(HDF_PATH)
        
    for data in DATASETS:
        print('\nreading the ts file {}'.format(data.name))
        df = data.load()
        file_path = HDF_PATH+data.name+EXTENSION

        print('wrinting {} on the path {}'.format(data.name,file_path))
        write_ts(df.loc['train'], file_path, 'train')
        write_ts(df.loc['test'], file_path, 'test')

