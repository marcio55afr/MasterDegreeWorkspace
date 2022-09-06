import wget
import time
import pandas as pd
import numpy as np

from sktime.benchmarking.data import UEADataset
from source.data.config import *

DATASET_NAMES = ['ECG5000', 'Worms', 'StarLightCurves']
EXTENSION = '.h5'
KEYS = ['train', 'test']


def save_all_datasets_as_hdf():
    if not os.path.exists(UNIVARIATE_TS_PATH):
        wget_all_datasets()

    if not os.path.exists(UNIVARIATE_HDF_PATH):
        os.mkdir(UNIVARIATE_HDF_PATH)

    uea_datasets = [UEADataset(path=UNIVARIATE_TS_PATH, name=name) for name in DATASET_NAMES]

    for data in uea_datasets:
        file_path = os.path.join(UNIVARIATE_HDF_PATH, data.name + EXTENSION)
        if os.path.isfile(file_path):
            continue

        print(f'\nReading ts file {data.name}')
        df = data.load()

        print(f'Writing {data.name} on the path {file_path}')
        write_ts(df.loc['train'], file_path, 'train')
        write_ts(df.loc['test'], file_path, 'test')


def write_ts(data: pd.DataFrame, path, key):
    data.index = range(data.shape[0])
    data.columns = ['data', 'target']
    data = data.astype({'target': np.int32})
    data.to_hdf(path, key=key)


def wget_all_datasets():
    print("Downloading timeseries datasets...\nit can takes a few minutes")
    univariate_ts_zip = wget.download(UNIVARIATE_TS_LINK,
                                      out=os.path.dirname(os.path.abspath(__file__)))
    time.sleep(1)

    if not os.path.exists(univariate_ts_zip):
        raise f'Download failed!\nos.path.isfile({univariate_ts_zip} is equal to {os.path.isfile(univariate_ts_zip)})'
    else:
        print('Download completed!')

    print("Unzipping the archive...\n")
    with zipfile.ZipFile(univariate_ts_zip, 'r') as zip_ref:
        zip_ref.extractall(UNIVARIATE_TS_PATH)
    os.remove(univariate_ts_zip)

    print("It's done!\n")
