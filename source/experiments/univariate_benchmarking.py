# -*- coding: utf-8 -*-

# import required functions and classes
import os
import sktime
import warnings

from sklearn.metrics import accuracy_score, f1_score, log_loss, balanced_accuracy_score
from sktime.benchmarking.data import UEADataset, make_datasets
from sktime.benchmarking.evaluation import Evaluator
from sktime.benchmarking.metrics import PairwiseMetric, AggregateMetric
from sktime.benchmarking.orchestration import Orchestrator
from sktime.benchmarking.results import HDDResults
from sktime.benchmarking.strategies import TSCStrategy
from sktime.benchmarking.tasks import TSCTask
from sktime.classification.dictionary_based import (
    ContractableBOSS,
    WEASEL
)
from sktime.classification.shapelet_based import (
    MrSEQLClassifier
)
from sktime.series_as_features.model_selection import PresplitFilesCV
import time

from datasets.univariate.config import DATASET_NAMES
from search_technique import SearchTechnique


#warnings.simplefilter(action='ignore', category=UserWarning)

# set up paths to data and results folder
DATA_PATH = os.path.join(os.path.abspath(os.getcwd()), "datasets/univariate/")
RESULTS_PATH = "results/"

#datasets = [
#    UEADataset(path=DATA_PATH, name="PenDigits")
#]

# Alternatively, we can use a helper function to create them automatically
names = DATASET_NAMES
datasets = [UEADataset(path=DATA_PATH, name=name) for name in names]

tasks = [TSCTask(target="target") for _ in range(len(datasets))]

# Specify learning strategies
strategies = [
    #TSCStrategy(SearchTechnique(), name="st_MeanTfidfPerClass_4r_RemoveBestResolutions"),
    TSCStrategy(ContractableBOSS(), name="CBOSS")
]


# Specify results object which manages the output of the benchmarking
results = HDDResults(path=RESULTS_PATH)

# run orchestrator
orchestrator = Orchestrator(
    datasets=datasets,
    tasks=tasks,
    strategies=strategies,
    cv=PresplitFilesCV(),
    results=results,
)
orchestrator.fit_predict(save_fitted_strategies=False,
                         overwrite_predictions=False, verbose=True)

evaluator = Evaluator(results)
print(evaluator.fit_runtime())

metrics = [PairwiseMetric(func=log_loss, name="log loss", proba=True, labels=True,)]
metrics += [PairwiseMetric(func=accuracy_score, name="accuracy")]
#metrics += [PairwiseMetric(func=balanced_accuracy_score, name="balanced accuracy")]
metrics += [AggregateMetric(func=f1_score, name="f1 score", labels=True, average='micro')]
for metric in metrics:
    metrics_by_strategy = evaluator.evaluate(metric=metric)
print(metrics_by_strategy)


























