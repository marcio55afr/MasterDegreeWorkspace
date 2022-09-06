import pandas as pd

from source.data.config import *
from source.utils.handlers.ts_handler import save_all_datasets_as_hdf

HDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hdf\\")
EXTENSION = '.h5'


def get_dataset(dataset_name) -> tuple:
    path = HDF_PATH + dataset_name + EXTENSION
    train = pd.read_hdf(path, key='train')
    test = pd.read_hdf(path, key='test')
    return train.data, train.target, test.data, test.target


def get_dataset_train(dataset_name) -> tuple:
    path = HDF_PATH + dataset_name + EXTENSION
    train = pd.read_hdf(path, key='train')
    return train.data, train.target


def get_dataset_test(dataset_name) -> tuple:
    path = HDF_PATH + dataset_name + EXTENSION
    test = pd.read_hdf(path, key='test')
    return test.data, test.target


def _check_dataset(dataset_name):
    path = HDF_PATH + dataset_name + EXTENSION
    if os.path.isfile(path):
        return

    if dataset_name not in DATASET_NAMES:
        raise "the dataset %s is unknown and cannot be read"

    if not os.path.isfile(path):
        save_all_datasets_as_hdf()
