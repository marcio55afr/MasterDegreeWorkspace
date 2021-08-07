import os
from sktime.benchmarking.data import UEADataset
import pandas as pd
import numpy as np

READING_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Univariate_ts\\")
WRITING_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hdf\\")

DATASET_NAMES = ['Worms', 'StarLightCurves', 'ECG5000']
DATASETS = [UEADataset(path=READING_PATH, name=name) for name in DATASET_NAMES]

def get_dataframe_from_tsfiles():
    for data in DATASETS:
        print('\nreading the ts file {}'.format(data.name))
        df = data.load()
        file_path = WRITING_PATH+data.name+'.h5'

        print('wrinting {} on the path {}'.format(data.name,file_path))
        write_ts(df.loc['train'], file_path, 'train')
        write_ts(df.loc['test'], file_path, 'test')

def write_ts( data: pd.DataFrame, path, key):
    data.index = range(data.shape[0])
    data.columns = ['data','target']
    data = data.astype({'target': np.int32})
    data.to_hdf(path, key=key)
