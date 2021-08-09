

"""
This is a experiment to proccess a dataset and get
some knowledge about the number of ngram words created
and its importance.

"""
import sys
sys.path.append('C:\\Users\\danie\\Documents\\Marcio\\MasterDegreeWorkspace\\source')
from experiments.database.ts_handler import get_dataset

train, test = get_dataset('ECG5000')

