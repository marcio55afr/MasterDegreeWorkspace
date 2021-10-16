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

from source.utils.cd_diagram import draw_cd_diagram
from datasets.univariate.config import DATASET_NAMES, LARGER_DATASETS_NAMES
from source.technique import (
    SearchTechnique,
    SearchTechniqueCV,
    SearchTechnique_CV_RFSF,
    SearchTechnique_SG,
    SearchTechnique_SG_RR,
    SearchTechnique_WS,
    SearchTechnique_MR_WS,
    SearchTechnique_DWS,
    SearchTechnique_5WS,
    SearchTechnique_KWS,
    SearchTechnique_SG_CLF,
    SearchTechnique_MR
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
names = DATASET_NAMES[2:]
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
        SearchTechniqueCV(discretization="SFA", feature_selection=True,
                          n_words=200, scoring='accuracy',
                          max_window_length = 1, max_num_windows = 20,
                          random_state=random_state),
        name="ST_CV_SFA_FS_l100"),
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

strategies_V1_clf = [
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '01',
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RandomForest_MF40"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RandomForest"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=10,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RandomForest_nw10"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=200,
                               random_selection=True,
                               rand_words=10,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RandomForest_rand10"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               ending_selection=True,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_DS_RandomForest"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               max_num_windows = 50,
                               n_words=None,
                               total_n_words = 200,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RandomForest_w50"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '03',
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RandomForest_Entropy"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '04',
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RandomForest_Entropy_L2"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '10',
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_SVC_rbf"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '11',
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_SVC_poly3"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '12',
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_SVC_poly5"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '13',
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_SVC_sigmoid"),    
    ]

strategies_V1 = [
    TSCStrategy_proba(
        SearchTechnique_KWS(K=1, discretization="SFA", n_words_variable=False,
                            ascending=False, random_state=random_state),
        name="ST_KWS_K1_Equal"), 
    TSCStrategy_proba(
        SearchTechnique_KWS(K=2, discretization="SFA", n_words_variable=False,
                            ascending=False, random_state=random_state),
        name="ST_KWS_K2_Equal"), 
    TSCStrategy_proba(
        SearchTechnique_KWS(K=5, discretization="SFA", n_words_variable=False,
                            ascending=False, random_state=random_state),
        name="ST_KWS_K5_Equal"), 
    TSCStrategy_proba(
        SearchTechnique_KWS(K=10, discretization="SFA", n_words_variable=False,
                            ascending=False, random_state=random_state),
        name="ST_KWS_K10_Equal"), 
    TSCStrategy_proba(
        SearchTechnique_KWS(K=20, discretization="SFA", n_words_variable=False,
                            ascending=False, random_state=random_state),
        name="ST_KWS_K20_Equal"),    
    TSCStrategy_proba(
        SearchTechnique_KWS(K=10, discretization="SFA", ascending=False,
                            n_words_variable=False,
                            random_selection = True, random_state=random_state),
        name="ST_KWS_K10_Equal_RS"), 
    TSCStrategy_proba(
        SearchTechnique_KWS(K=10, discretization="SFA",
                            ascending=False, random_state=random_state),
        name="ST_KWS_K10_Declined"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=20, discretization="SFA",
                            ascending=False, random_state=random_state),
        name="ST_KWS_K20_Declined"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=10, discretization="SFA", ascending=False,
                            random_selection = True, random_state=random_state),
        name="ST_KWS_K10_Declined_RS"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               discretization="SFA", 
                               ending_selection=True,
                               random_state=random_state),
        name="ST_SG_EndingSelection"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               discretization="SFA", 
                               random_selection=True,
                               random_state=random_state),
        name="ST_SG_RandomSelection"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=10,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_LowSelection_nw10"),  
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               max_num_windows=40,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_w40"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=10,
                               max_num_windows=40,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_LowSelection_nw10_w40"),
    ]



strategy_ST_WS = [
    TSCStrategy_proba(
        SearchTechnique_WS(discretization="SFA", method='mean',
                           n_best_windows = 1,
                           random_state=random_state),
        name="ST_SG_WS_RF_mean"),
    TSCStrategy_proba(
        SearchTechnique_WS(discretization="SFA", method='mean',
                           n_best_windows = 1,
                           random_state=random_state),
        name="ST_SG_WS_mean"),
    TSCStrategy_proba(
        SearchTechnique_WS(discretization="SFA", method='max',
                           n_best_windows = 1,
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

# ST_V1_WS_Test
strategies_WS_Test = [
    TSCStrategy_proba(
        SearchTechnique_DWS(discretization="SFA", random_state=random_state),
        name="ST_DWS"),
    TSCStrategy_proba(
        SearchTechnique_DWS(discretization="SFA", random_state=random_state),
        name="ST_DWS_full"),
    TSCStrategy_proba(
        SearchTechnique_DWS(discretization="SFA", random_state=random_state),
        name="ST_DWS_LRliblinear"),
    TSCStrategy_proba(
        SearchTechnique_DWS(discretization="SFA", random_state=random_state),
        name="ST_DWS_LRliblinear_balanced"),
    TSCStrategy_proba(
        SearchTechnique_DWS(discretization="SFA", random_state=random_state),
        name="ST_DWS_RF_est1000"),
    TSCStrategy_proba(
        SearchTechnique_DWS(discretization="SFA", random_state=random_state),
        name="ST_DWS_RFBalanced"),
    TSCStrategy_proba(
        SearchTechnique_DWS(discretization="SFA", random_state=random_state),
        name="ST_DWS_RFBalanced_est1000"),
    TSCStrategy_proba(
        SearchTechnique_DWS(discretization="SFA", random_state=random_state),
        name="ST_DWS_RFBalanced_maxF5"),
    TSCStrategy_proba(
        SearchTechnique_DWS(discretization="SFA", random_state=random_state),
        name="ST_DWS_RFBalanced_leaf2"),
    TSCStrategy_proba(
        SearchTechnique_DWS(discretization="SFA", random_state=random_state),
        name="ST_DWS_RFBalanced_maxf4_est1000"),
    TSCStrategy_proba(
        SearchTechnique_DWS(discretization="SFA", random_state=random_state),
        name="ST_DWS_SVC"),
    TSCStrategy_proba(
        SearchTechnique_DWS(discretization="SFA", random_state=random_state),
        name="ST_DWS_SVC_balanced"),
    TSCStrategy_proba(
        SearchTechnique_DWS(discretization="SFA", random_state=random_state),
        name="ST_DWS_NuSVC"),
    TSCStrategy_proba(
        SearchTechnique_5WS(discretization="SFA", random_state=random_state),
        name="ST_5WS"),
    TSCStrategy_proba(
        SearchTechnique_5WS(discretization="SFA", random_state=random_state),
        name="ST_5WS_Equal"),
    TSCStrategy_proba(
        SearchTechnique_5WS(discretization="SFA", random_state=random_state),
        name="ST_5WS_Rand"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=10, discretization="SFA",
                            ascending=False, random_state=random_state),
        name="ST_KWS_K10_Declined"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=10, discretization="SFA", n_words=200,
                            ascending=False,
                            random_state=random_state),
        name="ST_KWS_nw200_K10_Declined"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=10, discretization="SFA", n_words=100,
                            ascending=False,
                            random_state=random_state),
        name="ST_KWS_nw100_K10_Declined"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=10, discretization="SFA", n_words=100,
                            ascending=False,
                            random_state=random_state),
        name="ST_KWS_nw100_K10_Declined_150"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=10, discretization="SFA", n_words=200,
                            ascending=True,
                            random_state=random_state),
        name="ST_KWS_nw200_K10_Ascending"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=10, discretization="SFA", n_words=100,
                            ascending=True,
                            random_state=random_state),
        name="ST_KWS_nw100_K10_Ascending"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=15, discretization="SFA",
                            ascending=False, random_state=random_state),
        name="ST_KWS_K15_Declined"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=20, discretization="SFA",
                            ascending=False, random_state=random_state),
        name="ST_KWS_K20_Declined"), 
    TSCStrategy_proba(
        SearchTechnique_KWS(K=10, discretization="SFA", n_words_variable=False,
                            ascending=False, random_state=random_state),
        name="ST_KWS_K10_Equal"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=15, discretization="SFA", n_words_variable=False,
                            ascending=False, random_state=random_state),
        name="ST_KWS_K15_Equal"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=20, discretization="SFA", n_words_variable=False,
                            ascending=False, random_state=random_state),
        name="ST_KWS_K20_Equal"), 
    TSCStrategy_proba(
        SearchTechnique_KWS(K=10, discretization="SFA", random_top_words = True,
                            ascending=False, random_state=random_state),
        name="ST_KWS_K10_RandDeclined"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=15, discretization="SFA", random_top_words = True,
                            ascending=False, random_state=random_state),
        name="ST_KWS_K15_RandDeclined"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=20, discretization="SFA", random_top_words = True,
                            ascending=False,
                            random_state=random_state),
        name="ST_KWS_K20_RandDeclined"), 
    TSCStrategy_proba(
        SearchTechnique_KWS(K=10, discretization="SFA", ascending=False,
                            random_selection = True, random_state=random_state),
        name="ST_KWS_K10_Declined_RF"),
    ]

'''hello'''
# ST_V1_SG_Test
strategies_SG_Test = [
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '01',
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RF_MF40"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RF"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=5,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RF_nw5"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=10,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RF_nw10"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=200,
                               random_selection=True,
                               rand_words=10,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RF_nw200_rand10"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=200,
                               random_selection=True,
                               rand_words=20,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RF_nw200_rand20"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               max_num_windows = 50,
                               n_words=None,
                               total_n_words = 200,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RF_nw4_w50"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               max_num_windows = 20,
                               n_words=None,
                               total_n_words = 200,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RF_nw10_w20"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               max_num_windows = 30,
                               n_words=None,
                               total_n_words = 100,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RF_nw3_w30"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               max_num_windows = 20,
                               n_words=None,
                               total_n_words = 100,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RF_nw5_w10"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               max_num_windows = 100,
                               n_words=None,
                               total_n_words = 200,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RF_nw2_w100"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=10,
                               discretization="SAX", 
                               random_state=random_state),
        name="ST_SG_RF_SAX_nw10"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=15,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RF_nw15"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=5,
                               max_num_windows = 60,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RF_nw5_w60"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '03',
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RF_Entropy"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '04',
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RF_Entropy_Leaf2"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '10',
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_SVC_rbf"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '11',
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_SVC_poly3"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '12',
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_SVC_poly5"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '13',
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_SVC_sigmoid"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '04',
                               ending_selection=True,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_DS_RF_Entropy_Leaf2"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '04',
                               ending_selection=True,
                               random_selection=True,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_DRS_RF_Entropy_Leaf2")
    ]

# ST_V1_FULL
strategies_V1_FULL = [
    TSCStrategy_proba(
        SearchTechnique_WS(discretization="SFA", random_state=random_state),
        name="ST_SG_SFA"),
    TSCStrategy_proba(
        SearchTechnique_WS(discretization="SFA", method='mean',
                           random_state=random_state),
        name="ST_SG_WS_mean"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=10,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RF_nw10"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=10, discretization="SFA",
                            ascending=False, random_state=random_state),
        name="ST_KWS_K10_Declined"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               max_num_windows = 20,
                               n_words=None,
                               total_n_words = 200,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RF_nw10_Variable"),
]


# ST_V2
strategies_MR = [
    TSCStrategy_proba(
        SearchTechnique_MR(random_state=random_state),
        name="ST_MR"),
    TSCStrategy_proba(
        SearchTechnique_MR(random_state=random_state),
        name="ST_MR_saxlimit"),
    TSCStrategy_proba(
        SearchTechnique_MR(total_n_words = 400, random_state=random_state),
        name="ST_MR_saxlimit_tw400"),
    TSCStrategy_proba(
        SearchTechnique_MR(total_n_words = 800, random_state=random_state),
        name="ST_MR_saxlimit_tw800"),
    TSCStrategy_proba(
        SearchTechnique_MR(random_selection = True, total_n_words = 800,
                           random_state=random_state),
        name="ST_MR_saxlimit_randselect_tw800"),
    TSCStrategy_proba(
        SearchTechnique_MR(max_sfa_windows = 40, max_sax_windows=4,
                           random_state=random_state),
        name="ST_MR_40_4"),
    TSCStrategy_proba(
        SearchTechnique_MR(max_sfa_windows = 40, max_sax_windows=4,
                           total_n_words = 400,
                           random_state=random_state),
        name="ST_MR_40_4_tw400"),
    TSCStrategy_proba(
        SearchTechnique_MR(random_selection = True,
                           random_state=random_state),
        name="ST_MR_RandSelect"),
    TSCStrategy_proba(
        SearchTechnique_MR(randomize_best_words = True, total_n_words = 400,
                           random_state=random_state),
        name="ST_MR_RandWords_tw400"),
    TSCStrategy_proba(
        SearchTechnique_MR(max_sfa_windows = 40, max_sax_windows=4,
                           randomize_best_words = True, total_n_words = 400,
                           random_state=random_state),
        name="ST_MR_40_4_RandWords_tw400"),
    
    
    ]

strategy = strategies_V1_FULL
result_path = RESULTS_PATH + "ST_V1_Full"

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

scores = evaluator.get_all_datasets_scores('Accuracy', accuracy_score)
print(scores.groupby('strategy_name').mean())
#draw_cd_diagram(df_perf=scores, image_path=result_path, title='Accuracy', labels=True)
#input('end?')


#metrics = [PairwiseMetric(func=log_loss, name="Log Loss", proba=True, labels=True,)]
metrics = [PairwiseMetric(func=accuracy_score, name="Accuracy")]
#metrics += [PairwiseMetric(func=balanced_accuracy_score, name="Balanced Accuracy")]
metrics += [AggregateMetric(func=roc_auc_score, name="AUC ROC", proba=True, labels=True, multi_class='ovr',)]
#metrics += [AggregateMetric(func=f1_score, name="F1 Score", labels=True, average='micro')]
for metric in metrics:
    metrics_by_strategy = evaluator.evaluate(metric=metric)

#evaluator.plot_critical_difference_diagram('Accuracy')
#evaluator.plot_critical_difference_diagram('AUC ROC')
    
metrics_by_strategy = metrics_by_strategy.set_index('strategy')
metrics_by_strategy = pd.concat([metrics_by_strategy, runtime], axis=1)
print(metrics_by_strategy.iloc[:,[0,2,4]])

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







