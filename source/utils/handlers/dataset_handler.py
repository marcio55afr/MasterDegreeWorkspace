import numpy as np
import pandas as pd

from source.data.config import *
from source.utils import TsHandler


class DatasetHandler:

    smallest_dataset = 'SmoothSubspace'

    @classmethod
    def get_data_info(cls):
        return pd.read_csv(DATA_INFO, index_col=0)

    @classmethod
    def get_all_names(cls):
        return DATASET_NAMES

    @classmethod
    def get_longest_datasets(cls):
        return LONGEST_DATASETS

    @classmethod
    def get_widest_datasets(cls):
        return WIDEST_DATASETS

    @classmethod
    def get_train_data(cls, dataset_name):
        train_x, train_y = cls.get_split_dataset(dataset_name, 'train')
        train_x = cls.parse_to_array(train_x)
        train_y = np.asarray(train_y)
        return train_x, train_y

    @classmethod
    def get_test_data(cls, dataset_name):
        test_x, test_y = cls.get_split_dataset(dataset_name, 'test')
        test_x = cls.parse_to_array(test_x)
        test_y = np.asarray(test_y)
        return test_x, test_y

    @classmethod
    def get_split_data(cls, dataset_name) -> tuple[np.ndarray]:
        train_x, train_y = cls.get_train_data(dataset_name)
        test_x, test_y = cls.get_test_data(dataset_name)
        return train_x, train_y, test_x, test_y

    @classmethod
    def get_split_sample(cls, n, dataset_name):
        train_x, train_y, test_x, test_y = cls.get_split_data(dataset_name)
        sample_train_x = np.concatenate((train_x[:n], train_x[-n:]))
        sample_train_y = np.concatenate((train_y[:n], train_y[-n:]))
        sample_test_x = np.concatenate((test_x[:n], test_x[-n:]))
        sample_test_y = np.concatenate((test_y[:n], test_y[-n:]))

        return sample_train_x, sample_train_y, sample_test_x, sample_test_y

    @classmethod
    def get_split_dataset(cls, dataset_name, split=None) -> tuple:
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
        TsHandler.download_all_ts_and_transform()

    @classmethod
    def parse_to_dataframe(cls, data):
        aux = [pd.DataFrame([row]) for row in data]
        df = pd.concat(aux, axis=0, ignore_index=True)
        return df

    @classmethod
    def parse_to_array(cls, data):
        aux = [row.values for row in data]
        arr = np.asarray(aux)
        return arr

    @classmethod
    def _check_dataset(cls, dataset_name):
        path = cls._get_dataset_path(dataset_name)

        if dataset_name not in DATASET_NAMES:
            raise f"the dataset {dataset_name} is not in our available data set."

        if os.path.isfile(path):
            return

        if not os.path.isfile(path):
            TsHandler.download_all_ts_and_transform()
            if not os.path.isfile(path):
                raise f"Error downloading the dataset {dataset_name}."

        return

    @classmethod
    def _get_dataset_path(cls, dataset_name):
        path = UNIVARIATE_HDF_PATH + dataset_name + EXTENSION
        return path
