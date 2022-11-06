import pandas as pd

from source.data.config import *
from source.utils.handlers.ts_handler import download_all_ts_and_transform


class DatasetHandler:

    @classmethod
    def get_dataset(cls, dataset_name, split=None) -> tuple:
        """
            Check for dataset names in the file source/data/config.py
        """
        cls._check_dataset(dataset_name)
        path = cls._get_dataset_path(dataset_name)

        if split == 'train':
            train = pd.read_hdf(path, key='train')
            return train.data, train.target

        elif split == 'test':
            test = pd.read_hdf(path, key='test')
            return test.data, test.target

        train = pd.read_hdf(path, key='train')
        test = pd.read_hdf(path, key='test')
        return train.data, train.target, test.data, test.target

    @classmethod
    def setup_datasets(cls):
        download_all_ts_and_transform()

    @classmethod
    def _check_dataset(cls, dataset_name):
        path = cls._get_dataset_path(dataset_name)

        if dataset_name not in DATASET_NAMES:
            raise f"the dataset {dataset_name} is not in our available data set."

        if os.path.isfile(path):
            return

        if not os.path.isfile(path):
            download_all_ts_and_transform()
            if not os.path.isfile(path):
                raise f"Error downloading the dataset {dataset_name}."

        return

    @classmethod
    def _get_dataset_path(cls, dataset_name):
        path = UNIVARIATE_HDF_PATH + dataset_name + EXTENSION
        return path
