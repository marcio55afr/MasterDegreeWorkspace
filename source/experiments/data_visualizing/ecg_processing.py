"""
This is a experiment to proccess a dataset and get
some knowledge about the number of ngram words created
and its importance.

"""

import sys
sys.path.append('C:\\Users\\marci\\Desktop\\MasterDegreeWorkspace\\source')
from experiments.database.ts_handler import get_dataset
from utils.resolution_handler import ResolutionHandler

train, test = get_dataset('ECG5000')

ResolutionHandler.create_matrix(train.iloc[0,0].size)