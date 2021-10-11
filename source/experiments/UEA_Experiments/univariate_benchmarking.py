# -*- coding: utf-8 -*-
import sys
sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace/source')
sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace')
# import required functions and classes
import os
import sktime
import warnings
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, log_loss, balanced_accuracy_score, roc_auc_score
from sktime.benchmarking.data import UEADataset, make_datasets
from sktime.benchmarking.evaluation import Evaluator
from sktime.benchmarking.metrics import PairwiseMetric, AggregateMetric
from sktime.benchmarking.orchestration import Orchestrator
from sktime.benchmarking.results import HDDResults
from sktime.benchmarking.strategies import TSCStrategy, TSCStrategy_proba
from sktime.benchmarking.tasks import TSCTask
from sktime.classification.dictionary_based import (
    ContractableBOSS,
    WEASEL
)
from sktime.classification.shapelet_based import (
    MrSEQLClassifier
)
from sktime.series_as_features.model_selection import PresplitFilesCV

from datasets.univariate.config import DATASET_NAMES, LARGER_DATASETS_NAMES
from source.technique import (
    SearchTechnique,
    SearchTechniqueCV,
    SearchTechnique_CV_RFSF,
    SearchTechnique_SG,
    SearchTechnique_SG_RR,
    SearchTechnique_WS
)

warnings.simplefilter(action='ignore', category=UserWarning)

# set up paths to data and results folder
DATA_PATH = os.path.join(os.path.abspath(os.getcwd()), "datasets/univariate/")
RESULTS_PATH = "results/"
EVALUATION_FILE = RESULTS_PATH + "evaluation_v"

#datasets = [
#    UEADataset(path=DATA_PATH, name="PenDigits")
#]

# Alternatively, we can use a helper function to create them automatically
names = DATASET_NAMES
names = LARGER_DATASETS_NAMES
datasets = [UEADataset(path=DATA_PATH, name=name) for name in names]

tasks = [TSCTask(target="target") for _ in range(len(datasets))]

# Specify learning strategies

random_state = 27
strategies = []
results_paths = []

strategies_benchmark = [
    TSCStrategy_proba(ContractableBOSS(), name="CBOSS")        
    ]

strategies_ST_CV = [
    TSCStrategy_proba(
        SearchTechniqueCV(discretization="SFA", random_state=random_state,
                          scoring='accuracy'),
        name="ST_CV_SFA"),
    TSCStrategy_proba(
        SearchTechniqueCV(discretization="SAX", random_state=random_state,
                          scoring='accuracy'),
        name="ST_CV_SAX"),
    TSCStrategy_proba(
        SearchTechniqueCV(discretization="SAX", feature_selection=True,
                          n_words=200, scoring='accuracy',
                          random_state=random_state),
        name="ST_CV_SAX_FS"),
    TSCStrategy_proba(
        SearchTechniqueCV(discretization="SFA", feature_selection=True,
                          n_words=200, scoring='accuracy',
                          random_state=random_state),
        name="ST_CV_SFA_FS"),
    TSCStrategy_proba(
        SearchTechnique_CV_RFSF(discretization="SAX", n_words=200,
                                random_state=random_state),
        name="ST_CV_SAX_RFFS"),
    TSCStrategy_proba(
        SearchTechnique_CV_RFSF(discretization="SFA", n_words=200,
                                random_state=random_state),
        name="ST_CV_SFA_RFFS"),
    TSCStrategy_proba(
        SearchTechnique_CV_RFSF(discretization="SFA", n_words=200,
                                max_window_length = 1., max_num_windows = 20,
                                random_state=random_state),
        name="ST_CV_SFA_RFFS_l100"),
    TSCStrategy_proba(
        SearchTechnique_CV_RFSF(discretization="SFA", n_words=200,
                                max_window_length = .5, max_num_windows = 100,
                                random_state=random_state),
        name="ST_CV_SFA_RFFS_w100"),
    TSCStrategy_proba(
        SearchTechnique_CV_RFSF(discretization="SFA", n_words=200,
                                max_window_length = 1., max_num_windows = 100,
                                random_state=random_state),
        name="ST_CV_SFA_RFFS_l100_w100"),
    TSCStrategy_proba(
        SearchTechniqueCV(discretization="SFA", feature_selection=True,
                          n_words=200, scoring='accuracy',
                          max_window_length = 1, max_num_windows = 20,
                          random_state=random_state),
        name="ST_CV_SFA_FS_l100"),
    TSCStrategy_proba(
        SearchTechniqueCV(discretization="SFA", feature_selection=True,
                          n_words=200, scoring='accuracy',
                          max_window_length = .5, max_num_windows = 100,
                          random_state=random_state),
        name="ST_CV_SFA_FS_w100"),
    TSCStrategy_proba(
        SearchTechniqueCV(discretization="SFA", feature_selection=True,
                          n_words=200, scoring='accuracy',
                          max_window_length = 1, max_num_windows = 100,
                          random_state=random_state),
        name="ST_CV_SFA_FS_l100_w100"),
    TSCStrategy_proba(
        SearchTechniqueCV(discretization="SFA", feature_selection=True,
                          n_words=200, scoring='accuracy',
                          remove_repeat_words=True,
                          random_state=random_state),
        name="ST_CV_SFA_FS_RRW"),
]
#strategies.append(strategies_ST_CV)
#results_paths.append(RESULTS_PATH + "ST_V0")



strategies_ST_SG= [
    TSCStrategy_proba(
        SearchTechnique_SG(discretization="SAX", random_state=random_state),
        name="ST_SG_SAX"),
    TSCStrategy_proba(
        SearchTechnique_SG(discretization="SFA", random_state=random_state),
        name="ST_SG_SFA"),
    TSCStrategy_proba(
        SearchTechnique_SG(discretization="SFA", random_state=random_state,
                           ending_selection=True),
        name="ST_SG_SFA_DS"),    
    TSCStrategy_proba(
        SearchTechnique_SG(discretization="SFA", random_state=random_state,
                           early_selection=False, ending_selection=True),
        name="ST_SG_SFA_EndS"),
    TSCStrategy_proba(
        SearchTechnique_SG(discretization="SFA", random_state=random_state,
                           ending_selection=True, random_selection=True),
        name="ST_SG_SFA_EarRandEndS"),
    TSCStrategy_proba(
        SearchTechnique_SG(discretization="SFA", random_state=random_state,
                           early_selection=False, ending_selection=True,
                           random_selection=True),
        name="ST_SG_SFA_RSFS"),
    TSCStrategy_proba(
        SearchTechnique_SG(discretization="SFA", random_state=random_state,
                           early_selection=True, ending_selection=True,
                           random_selection=True),
        name="ST_SG_SFA_RSDS"),
    TSCStrategy_proba(
        SearchTechnique_SG_RR(discretization="SFA",
                              random_state=random_state),
        name="ST_SG_SFA_RR"),
    ]

strategy_ST_WS = [
    TSCStrategy_proba(
        SearchTechnique_WS(discretization="SFA", method='mean',
                           random_state=random_state),
        name="ST_SG_WS_mean"),
    TSCStrategy_proba(
        SearchTechnique_WS(discretization="SFA", method='max',
                           random_state=random_state),
        name="ST_SG_WS_max"),
    TSCStrategy_proba(
        SearchTechnique_WS(discretization="SFA", method='mean',
                           n_best_windows = 3,
                           random_state=random_state),
        name="ST_SG_3WS_mean"),
    TSCStrategy_proba(
        SearchTechnique_WS(discretization="SFA", method='max',
                           n_best_windows = 3,
                           random_state=random_state),
        name="ST_SG_3WS_max"),
    TSCStrategy_proba(
        SearchTechnique_WS(discretization="SFA", method='mean',
                           n_best_windows = 2,
                           random_state=random_state),
        name="ST_SG_2WS_mean"),
    TSCStrategy_proba(
        SearchTechnique_WS(discretization="SFA", method='max',
                           n_best_windows = 2,
                           random_state=random_state),
        name="ST_SG_2WS_max"),
    TSCStrategy_proba(
        SearchTechnique_WS(discretization="SFA", method='mean',
                           n_best_windows = 2, n_words=100,
                           random_state=random_state),
        name="ST_SG_2WS_nw100_mean"),
    TSCStrategy_proba(
        SearchTechnique_WS(discretization="SFA", method='max',
                           n_best_windows = 2, n_words=100,
                           random_state=random_state),
        name="ST_SG_2WS_nw100_max"),
]

strategy = [
    TSCStrategy_proba(
        SearchTechnique_WS(discretization="SFA", random_state=random_state),
        name="ST_SG_SFA"),
    TSCStrategy_proba(
        SearchTechnique_WS(discretization="SFA", method='mean',
                           random_state=random_state),
        name="ST_SG_WS_mean"),
]


#strategy = strategies_ST_SG
result_path = RESULTS_PATH + "ST_V1_FULL"

# Specify results object which manages the output of the benchmarking
results = HDDResults(path=result_path)

# run orchestrator
orchestrator = Orchestrator(
    datasets=datasets,
    tasks=tasks,
    strategies=strategy,
    cv=PresplitFilesCV(),
    results=results,
)
orchestrator.fit_predict(save_fitted_strategies=False,
                         overwrite_predictions=False, verbose=True)

evaluator = Evaluator(results)
runtime = evaluator.fit_runtime().groupby('strategy_name').mean()
print(runtime)

#metrics = [PairwiseMetric(func=log_loss, name="Log Loss", proba=True, labels=True,)]
metrics = [PairwiseMetric(func=accuracy_score, name="Accuracy")]
#metrics += [PairwiseMetric(func=balanced_accuracy_score, name="Balanced Accuracy")]
#metrics += [AggregateMetric(func=roc_auc_score, name="AUC ROC", proba=True, labels=True, multi_class='ovr',)]
#metrics += [AggregateMetric(func=f1_score, name="F1 Score", labels=True, average='micro')]
for metric in metrics:
    metrics_by_strategy = evaluator.evaluate(metric=metric)

#evaluator.plot_critical_difference_diagram('Accuracy')
#evaluator.plot_critical_difference_diagram('AUC ROC')
    
metrics_by_strategy = metrics_by_strategy.set_index('strategy')
metrics_by_strategy = pd.concat([metrics_by_strategy, runtime], axis=1)
print(metrics_by_strategy.iloc[:,[0,2]])

fig, ax = plt.subplots(figsize=[8,6], dpi=200)

n_strategy = metrics_by_strategy.shape[0]
colors = np.arange(n_strategy)
scatter = ax.scatter(metrics_by_strategy['fit_runtime'],
                     metrics_by_strategy['Accuracy_mean'],
                     label=metrics_by_strategy.index.values,
                     c=colors, cmap='viridis')

handles, labels = scatter.legend_elements(prop="colors")
legend1 = ax.legend(handles, metrics_by_strategy.index.values ,
                    loc="center right", title="Classes",
                    fancybox=True,
                    title_fontsize=14,
                    borderpad=1.0)
ax.add_artist(legend1)
ax.grid(True)

plt.show()







