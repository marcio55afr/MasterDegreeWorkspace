import unittest
import random
from source.utils import DatasetHandler


class TestFunction_get_dataset(unittest.TestCase):

    def test_execution(self):
        data = DatasetHandler.get_split_dataset('SmoothSubspace')

        assert all([split is not None for split in data])
