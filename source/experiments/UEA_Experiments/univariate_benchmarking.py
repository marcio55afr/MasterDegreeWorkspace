# -*- coding: utf-8 -*-
import sys
sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace/source')
sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace')
sys.path.append('/home/marcio55agb/MasterDegreeWorkspace')
# import required functions and classes
import os
import sktime
import warnings
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, roc_auc_score#f1_score, log_loss, balanced_accuracy_score, 
from sktime.benchmarking.data import UEADataset, make_datasets
#from sktime.benchmarking.evaluation import Evaluator
#from sktime.benchmarking.metrics import PairwiseMetric, AggregateMetric
from sktime.benchmarking.orchestration import Orchestrator
#from sktime.benchmarking.results import HDDResults
#from sktime.benchmarking.strategies import TSCStrategy, TSCStrategy_proba
from sktime.benchmarking.tasks import TSCTask
from sktime.classification.dictionary_based import (
    ContractableBOSS,
    WEASEL
)
from sktime.classification.shapelet_based import (
    MrSEQLClassifier
)
from sktime.series_as_features.model_selection import PresplitFilesCV

#from sktime_changes import PairwiseMetric, AggregateMetric
from source.experiments.sktime_changes import Evaluator, HDDResults,TSCStrategy_proba

from source.utils import draw_cd_diagram, calculate_efficiency
from source.experiments.UEA_Experiments.datasets import DATASET_NAMES, LARGER_DATASETS_NAMES
from source.technique import (
    RandomClassifier,
    SearchTechnique,
    SearchTechniqueCV,
    SearchTechnique_CV_RFSF,
    SearchTechnique_KWS,
    SearchTechnique_SG_CLF,
    SearchTechnique_MD,
    SearchTechnique_Ngram,
    SearchTechnique_NgramResolution
)

warnings.simplefilter(action='ignore', category=UserWarning)

# set up paths to data and results folder
actual_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(actual_path, "datasets/Univariate_ts/")

SCORE_PATH =  "scores/"
RESULT_PATH = os.path.join(actual_path, "results/")

#datasets = [
#    UEADataset(path=DATA_PATH, name="PenDigits")
#]

# Alternatively, we can use a helper function to create them automatically
names = LARGER_DATASETS_NAMES
names = DATASET_NAMES
datasets = [UEADataset(path=DATA_PATH, name=name) for name in names]

tasks = [TSCTask(target="target") for _ in range(len(datasets))]

# Specify learning strategies

random_state = 27
strategies = []
results_paths = []

RANDOM_CLF_ACC_mean = 40.5545
RANDOM_CLF_ROC_AUC_mean = 49.5794

strategies_benchmark = [
    TSCStrategy_proba(ContractableBOSS(), name="CBOSS")        
    ]

strategies_random = [
    TSCStrategy_proba(
        RandomClassifier(random_state=None),
        name="RandomClassifier"
    )
]

strategies_V0 = [
    TSCStrategy_proba(
        SearchTechniqueCV(discretization="SFA", random_state=random_state,),
        name="ST_CV_SFA"),
    TSCStrategy_proba(
        SearchTechniqueCV(discretization="SFA", random_state=random_state,
                          max_window_length = 1.),
        name="ST_CV_SFA_l100"),
    TSCStrategy_proba(
        SearchTechniqueCV(discretization="SFA", feature_selection=True,
                          n_words=200,
                          random_state=random_state),
        name="ST_CV_SFA_FS"),
    TSCStrategy_proba(
        SearchTechnique_CV_RFSF(discretization="SFA", n_words=200,
                                random_state=random_state),
        name="ST_CV_SFA_RFFS"),
    TSCStrategy_proba(
        SearchTechniqueCV(discretization="SFA", feature_selection=True,
                          n_words=100,
                          random_state=random_state),
        name="ST_CV_SFA_FS_nw100"),
    TSCStrategy_proba(
        SearchTechniqueCV(discretization="SFA", feature_selection=True,
                          n_words=500,
                          random_state=random_state),
        name="ST_CV_SFA_FS_nw500"),
    TSCStrategy_proba(
        SearchTechnique_CV_RFSF(discretization="SFA", n_words=100,
                                random_state=random_state),
        name="ST_CV_SFA_RFFS_nw100"),
    TSCStrategy_proba(
        SearchTechniqueCV(discretization="SAX", random_state=random_state),
        name="ST_CV_SAX"),
    TSCStrategy_proba(
        SearchTechniqueCV(discretization="SAX", random_state=random_state,
                          max_window_length = 1.),
        name="ST_CV_SAX_l100"),
    TSCStrategy_proba(
        SearchTechniqueCV(discretization="SAX", feature_selection=True,
                          n_words=200,
                          random_state=random_state),
        name="ST_CV_SAX_FS"),
    TSCStrategy_proba(
        SearchTechnique_CV_RFSF(discretization="SAX", n_words=200,
                                random_state=random_state),
        name="ST_CV_SAX_RFFS"),
    TSCStrategy_proba(
        SearchTechniqueCV(discretization="SAX", feature_selection=True,
                          n_words=200,
                          random_state=random_state),
        name="ST_CV_SAX_FS_nw100"),
    TSCStrategy_proba(
        SearchTechniqueCV(discretization="SAX", feature_selection=True,
                          n_words=500,
                          random_state=random_state),
        name="ST_CV_SAX_FS_nw500"),
    TSCStrategy_proba(
        SearchTechnique_CV_RFSF(discretization="SAX", n_words=200,
                                random_state=random_state),
        name="ST_CV_SAX_RFFS_nw100"),
]

strategies_V0_FULL = [
    TSCStrategy_proba(
        SearchTechniqueCV(discretization="SFA", feature_selection=True,
                          n_words=100,
                          random_state=random_state),
        name="ST_CV_SFA_FS_nw100")
    ]

strategies_V1_clf = [
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '20',
                               n_words=100,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_LogisticRegression"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '20',
                               n_words=100,
                               random_selection=True,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RES_LogisticRegression"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '20',
                               n_words=10,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_LogisticRegression_nw10"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '21',
                               n_words=100,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_LogisticRegression_lib"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '21',
                               n_words=10,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_LogisticRegression_lib_nw10"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '01',
                               n_words=100,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RandomForest_MF40"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=100,
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
                               n_words=100,
                               rand_words=10,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RandomForest_rand10"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=100,
                               random_selection=True,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RES_RandomForest"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=100,
                               ending_selection=True,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_DS_RandomForest"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '03',
                               n_words=100,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RandomForest_Entropy"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '04',
                               n_words=100,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RandomForest_L2"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '10',
                               n_words=100,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_SVC_rbf"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '10',
                               n_words=100,
                               random_selection=True,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RES_SVC_rbf"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '10',
                               n_words=10,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_SVC_rbf_nw10"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '14',
                               n_words=100,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_SVC_rbf_balanced"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '14',
                               n_words=10,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_SVC_rbf_balanced_nw10"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '11',
                               n_words=100,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_SVC_poly3"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '11',
                               discretization="SFA", 
                               n_words = 10,
                               random_state=random_state),
        name="ST_SG_SVC_poly3_nw10"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '12',
                               n_words=100,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_SVC_poly5"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '13',
                               n_words=100,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_SVC_sigmoid"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '13',
                               n_words = 10,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_SVC_sigmoid_nw10"),
    ]


strategies_V1_sg = [
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=10,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_nw10_w20"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=20,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_nw20_w20"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=30,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_nw30_w20"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=10,
                               max_num_windows=15,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_nw10_w15"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=10,
                               max_num_windows=25,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_nw10_w25"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=20,
                               max_num_windows=15,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_nw20_w15"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=20,
                               max_num_windows=25,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_nw20_w25"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=None,
                               total_n_words = 200,
                               random_state=random_state),
        name="ST_SG_tw200"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=200,
                               discretization="SFA", 
                               ending_selection=True,
                               random_state=random_state),
        name="ST_SG_EndingSelection_nw200"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=20,
                               discretization="SFA", 
                               random_selection=True,
                               random_state=random_state),
        name="ST_SG_nw20_RandomSelection"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=40,
                               rand_words=20,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_rw20_of40"),
    ]
   

strategies_V1_kws = [ 
    TSCStrategy_proba(
        SearchTechnique_KWS(K=5, discretization="SFA",
                            func = "mean",
                            random_state=random_state),
        name="ST_KWS_K5_Equal"), 
    TSCStrategy_proba(
        SearchTechnique_KWS(K=10, discretization="SFA",
                            func = "mean",
                            random_state=random_state),
        name="ST_KWS_K10_Equal"), 
    TSCStrategy_proba(
        SearchTechnique_KWS(K=20, discretization="SFA",
                            func = "mean",
                            random_state=random_state),
        name="ST_KWS_K20_Equal"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=30, discretization="SFA",
                            func = "mean",
                            random_state=random_state),
        name="ST_KWS_K30_Equal"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=10, discretization="SFA",
                            random_state=random_state),
        name="ST_KWS_K10_Equal_max"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=20, discretization="SFA",
                            random_state=random_state),
        name="ST_KWS_K20_Equal_max"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=10, method='Declined', discretization="SFA",
                            func = "mean",
                            n_words=60,
                            inclination = 1.3,
                            random_state=random_state),
        name="ST_KWS_K10_Declined"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=20, method='Declined', discretization="SFA",
                            func = "mean",
                            n_words=40,
                            inclination = 1.15,
                            random_state=random_state),
        name="ST_KWS_K20_Declined"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=30, method='Declined', discretization="SFA",
                            func = "mean",
                            n_words=35,
                            inclination = 1.10,
                            random_state=random_state),
        name="ST_KWS_K30_Declined"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=20, method='Declined', discretization="SFA",
                            n_words=40,
                            inclination = 1.15,
                            random_state=random_state),
        name="ST_KWS_K20_Declined_max"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=30, method='Declined', discretization="SFA",
                            n_words=35,
                            inclination = 1.10,
                            random_state=random_state),
        name="ST_KWS_K30_Declined_max"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=10, discretization="SFA",
                            n_words=20,
                            random_top_words=True,
                            random_state=random_state),
        name="ST_KWS_K10_Equal_max_RES"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=10, discretization="SFA",
                            random_selection=True,
                            random_state=random_state),
        name="ST_KWS_K10_Equal_max_RS"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=30, method='Declined', discretization="SFA",
                            func = "mean",
                            n_words=70,
                            random_top_words=True,
                            inclination = 1.10,
                            random_state=random_state),
        name="ST_KWS_K30_Declined_RES"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=30, method='Declined', discretization="SFA",
                            func = "mean",
                            n_words=35,
                            random_selection=True,
                            inclination = 1.10,
                            random_state=random_state),
        name="ST_KWS_K30_Declined_RS"),
    ]


# ST_V1_FULL
strategies_V1_FULL = [
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=10,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_nw10_w20"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=30, discretization="SFA",
                            func = "mean",
                            random_state=random_state),
        name="ST_KWS_K30_Equal"),
]


# ST_V2
strategies_V2 = [
    TSCStrategy_proba(
        SearchTechnique_MD(random_state=random_state,
                           max_sfa_windows = 20,
                           max_sax_windows = 2,
                           n_sfa_words = 10,
                           n_sax_words = 20),
        name="ST_MD_w20_2_nw10_20"),
    TSCStrategy_proba(
        SearchTechnique_MD(random_state=random_state,
                           max_sfa_windows = 20,
                           max_sax_windows = 2,
                           n_sfa_words = 10,
                           n_sax_words = 40),
        name="ST_MD_w20_2_nw10_40"),
    TSCStrategy_proba(
        SearchTechnique_MD(random_state=random_state,
                           max_sfa_windows = 20,
                           max_sax_windows = 4,
                           n_sfa_words = 10,
                           n_sax_words = 20),
        name="ST_MD_w20_4_nw10_20"),
    TSCStrategy_proba(
        SearchTechnique_MD(random_state=random_state,
                           max_sfa_windows = 20,
                           max_sax_windows = 4,
                           n_sfa_words = 10,
                           n_sax_words = 40),
        name="ST_MD_w20_4_nw10_40"),
    TSCStrategy_proba(
        SearchTechnique_MD(random_state=random_state,
                           max_sfa_windows = 16,
                           max_sax_windows = 4,
                           n_sfa_words = 10,
                           n_sax_words = 20),
        name="ST_MD_w16_4_nw10_20"),
    ]
'''
    TSCStrategy_proba(
        SearchTechnique_MD(random_state=random_state,
                           max_sfa_windows = 16,
                           max_sax_windows = 4,
                           n_sfa_words = 10,
                           n_sax_words = 40),
        name="ST_MD_w16_4_nw10_40"),
    TSCStrategy_proba(
        SearchTechnique_MD(random_state=random_state,
                           max_sfa_windows = 16,
                           max_sax_windows = 2,
                           n_sfa_words = 20,
                           n_sax_words = 40),
        name="ST_MD_w16_2_nw10_40"),
    TSCStrategy_proba(
        SearchTechnique_MD(random_state=random_state,
                           max_sfa_windows = 20,
                           max_sax_windows = 1,
                           n_sfa_words = 20,
                           n_sax_words = 40),
        name="ST_MD_w20_1_nw20_40"),
    TSCStrategy_proba(
        SearchTechnique_MD(random_state=random_state,
                           randomize_best_words = True,
                           max_sfa_windows = 20,
                           max_sax_windows = 1,
                           n_sfa_words = 20,
                           n_sax_words = 40),
        name="ST_MD_RandWords_w20_1_nw20_40"),
    TSCStrategy_proba(
        SearchTechnique_MD(random_state=random_state,
                           randomize_best_words = True,
                           max_sfa_windows = 20,
                           max_sax_windows = 1,
                           n_sfa_words = 40,
                           n_sax_words = 80),
        name="ST_MD_RandWords_w20_1_nw40_80"),
    TSCStrategy_proba(
        SearchTechnique_MD(random_state=random_state,
                           random_selection = True,
                           max_sfa_windows = 20,
                           max_sax_windows = 1,
                           n_sfa_words = 20,
                           n_sax_words = 40),
        name="ST_MD_RandSelect_w20_1_nw20_40"),
    TSCStrategy_proba(
        SearchTechnique_MD(random_state=random_state,
                           max_sfa_windows = 20,
                           max_sax_windows = 1,
                           n_sfa_words = 10,
                           n_sax_words = 40),
        name="ST_MR_w20_1_nw10_40"),
    TSCStrategy_proba(
        SearchTechnique_MD(random_state=random_state,
                           max_sfa_windows = 20,
                           max_sax_windows = 1,
                           n_sfa_words = 10,
                           n_sax_words = 20),
        name="ST_MD_w20_1_nw10_20"),
    TSCStrategy_proba(
        SearchTechnique_MD(random_state=random_state,
                           max_sfa_windows = 20,
                           max_sax_windows = 1,
                           n_sfa_words = 20,
                           n_sax_words = 20),
        name="ST_MD_w20_1_nw20_20"),
]
'''

strategies_V2_FULL = [
    TSCStrategy_proba(
        SearchTechnique_MD(random_state=random_state,
                           max_sfa_windows = 20,
                           max_sax_windows = 1,
                           n_sfa_words = 10,
                           n_sax_words = 20),
        name="ST_MR_w20_1_nw10_20"),
    ]

strategies_V3 = [
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=10,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SFA_2gram"),
    TSCStrategy_proba(
        SearchTechnique_NgramResolution(N=5,
                                        random_state=random_state),
        name="ST_SFA_NgramResolution"),
    TSCStrategy_proba(
        SearchTechnique_NgramResolution(N=5,
                                        word_length=4,
                                        random_state=random_state),
        name="ST_SFA_NgramResolution_wl4"),
    TSCStrategy_proba(
        SearchTechnique_Ngram(only_sfa=True,
                              random_state=random_state),
        name="ST_SFA_Ngram"),
    TSCStrategy_proba(
        SearchTechnique_Ngram(N=3, only_sfa=True,
                              random_state=random_state),
        name="ST_SFA_Ngram_n3"),
    TSCStrategy_proba(
        SearchTechnique_Ngram(N=3, word_length=4, only_sfa=True,
                              random_state=random_state),
        name="ST_SFA_Ngram_n3_wl4"),
    TSCStrategy_proba(
        SearchTechnique_Ngram(N=3, word_length=4, only_sfa=True,
                              random_state=random_state),
        name="ST_SFA_Ngram_n3_wl4_svc"),
    TSCStrategy_proba(
        SearchTechnique_Ngram(random_state=random_state),
        name="ST_Ngram"),
    TSCStrategy_proba(
        SearchTechnique_Ngram(N=3,
                              random_state=random_state),
        name="ST_Ngram_n3"),
    TSCStrategy_proba(
        SearchTechnique_Ngram(N=1,
                              random_state=random_state),
        name="ST_Ngram_n1"),
    TSCStrategy_proba(
        SearchTechnique_Ngram(random_selection = True,
                              random_state=random_state),
        name="ST_Ngram_RS"),
    TSCStrategy_proba(
        SearchTechnique_Ngram(n_sfa_words = 20,
                              n_sax_words = 40,
                              randomize_best_words = True,
                              random_state=random_state),
        name="ST_Ngram_RBW"),
    TSCStrategy_proba(
        SearchTechnique_Ngram(n_sfa_words = 50,
                              n_sax_words = 100,
                              random_state=random_state),
        name="ST_Ngram_50_100"),
    ]

strategies_V4 = [
    TSCStrategy_proba(
        SearchTechnique(random_state=random_state),
        name="ST_MR"),    
    ]

strategy = strategies_V1_clf
variant = "ST_V1_clf/"

score_strategy_path = SCORE_PATH + variant

# Specify results object which manages the output of the benchmarking
results = HDDResults(path=score_strategy_path)

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

acc_scores = evaluator.get_all_datasets_scores('Accuracy', accuracy_score)
roc_scores = evaluator.get_all_datasets_scores('ROC AUC', roc_auc_score, probabilties=True, labels=True, multi_class='ovr')


result_strategy_path = RESULT_PATH + variant
if not os.path.isdir(result_strategy_path):
    os.mkdir(result_strategy_path)


#draw_cd_diagram(df_perf=acc_scores, image_path=result_path, title='Accuracy', labels=True)

acc_scores = acc_scores.groupby('strategy_name').mean()*100
roc_scores = roc_scores.groupby('strategy_name').mean()*100

score_results = pd.concat([acc_scores,roc_scores,runtime], axis=1)
score_results.columns = ['Accuracy mean', 'ROC AUC mean', 'fit runtime mean', 'predict runtime mean']
print(score_results.iloc[:,1:])

score_results['Accuracy efficency'] = calculate_efficiency(score_results.iloc[:,[0,2,3]],
                                                           RANDOM_CLF_ACC_mean)
score_results['ROC AUC efficency'] = calculate_efficiency(score_results.iloc[:,[1,2,3]],
                                                          RANDOM_CLF_ROC_AUC_mean)
score_results = score_results.sort_values('ROC AUC mean')
score_results = score_results.round(3)
score_results.to_csv(result_strategy_path+'scores.csv')
print(score_results.iloc[:,:2])
score_results = score_results.sort_values('ROC AUC efficency')
print(score_results.iloc[:,-2:])

fig, ax = plt.subplots(figsize=[8,6], dpi=200)

n_strategy = score_results.shape[0]
colors = np.arange(n_strategy)
scatter = ax.scatter(score_results['fit runtime mean'],
                     score_results['Accuracy mean'],
                     label=score_results.index.values,
                     c=colors, cmap='viridis')

handles, labels = scatter.legend_elements(prop="colors")
legend1 = ax.legend(handles, score_results.index.values ,
                    loc="center right", title="Classes",
                    fancybox=True,
                    title_fontsize=14,
                    borderpad=1.0)

ax.set_title('Accuracy effiency')
ax.set_xlabel('fit runtime mean')
ax.set_ylabel('accuracy mean')
ax.add_artist(legend1)
ax.grid(True)

plt.savefig(result_strategy_path+'acc_eficiency')


fig, ax = plt.subplots(figsize=[8,6], dpi=200)

n_strategy = score_results.shape[0]
colors = np.arange(n_strategy)
scatter = ax.scatter(score_results['fit runtime mean'],
                     score_results['ROC AUC mean'],
                     label=score_results.index.values,
                     c=colors, cmap='viridis')

handles, labels = scatter.legend_elements(prop="colors")
legend1 = ax.legend(handles, score_results.index.values ,
                    loc="center right", title="Classes",
                    fancybox=True,
                    title_fontsize=14,
                    borderpad=1.0)

ax.set_title('ROC AUC efficiency')
ax.set_xlabel('fit runtime mean')
ax.set_ylabel('roc auc mean')
ax.add_artist(legend1)
ax.grid(True)

plt.savefig(result_strategy_path+'roc_eficiency')




