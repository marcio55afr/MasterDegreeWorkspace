"""
This is a experiment to proccess a dataset and get
some knowledge about the number of ngram words created
and its importance.

"""

import sys
sys.path.append('C:\\Users\\marci\\Desktop\\MasterDegreeWorkspace\\source')
from experiments.database.ts_handler import get_dataset
from utils import ResolutionMatrix, ResolutionHandler, NgramExtractor
from transformations import MSAX

# handling data
train, test = get_dataset('ECG5000')
timeseries = train.data
labels = train.target

# initializating useful objects
rm = ResolutionMatrix(train.iloc[0,0].size)
wi, wo =  rm.get_windows_and_words()
tf = MSAX()


ts = timeseries.iloc[0]
word_seq = tf.transform(ts, wi, wo)
bob = NgramExtractor.get_bob(word_seq, rm.matrix)

#for ts in timeseries:
#    MSAX