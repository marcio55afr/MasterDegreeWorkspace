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
    SearchTechnique_CV_RSFS,
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

SCORE_PATH =  os.path.join(actual_path, "scores/")
RESULT_PATH = os.path.join(actual_path, "results/")

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
        SearchTechnique_CV_RSFS(discretization="SFA", n_words=200,
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
        SearchTechnique_CV_RSFS(discretization="SFA", n_words=100,
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
        SearchTechnique_CV_RSFS(discretization="SAX", n_words=200,
                                random_state=random_state),
        name="ST_CV_SAX_RFFS"),
    TSCStrategy_proba(
        SearchTechniqueCV(discretization="SAX", feature_selection=True,
                          n_words=100,
                          random_state=random_state),
        name="ST_CV_SAX_FS_nw100"),
    TSCStrategy_proba(
        SearchTechniqueCV(discretization="SAX", feature_selection=True,
                          n_words=500,
                          random_state=random_state),
        name="ST_CV_SAX_FS_nw500"),
    TSCStrategy_proba(
        SearchTechnique_CV_RSFS(discretization="SAX", n_words=100,
                                random_state=random_state),
        name="ST_CV_SAX_RFFS_nw100"),
]

strategies_V0_FULL = [
    TSCStrategy_proba(
        SearchTechnique_CV_RSFS(discretization="SFA", n_words=100,
                                random_state=random_state),
        name="ST_CV_SFA_RFFS_nw100"),
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
        name="ST_SG_RSFS_LogisticRegression"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '21',
                               n_words=100,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_LogisticRegression_lib"),
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
                               n_words=100,
                               random_selection = True,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RSFS_RandomForest"),
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
        name="ST_SG_RSFS_SVC_rbf"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '14',
                               n_words=100,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_SVC_rbf_balanced"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '11',
                               n_words=100,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_SVC_poly3"),
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
    ]


strategies_V1_sg = [
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=20,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_nw20_w20"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=50,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_nw50_w20"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=100,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_nw100_w20"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=200,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_nw200_w20"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=20,
                               max_num_windows = 30,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_nw20_w30"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=50,
                               max_num_windows = 30,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_nw50_w30"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=100,
                               max_num_windows = 30,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_nw100_w30"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=200,
                               max_num_windows = 30,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_nw200_w30"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=50,
                               max_num_windows = 10,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_nw50_w10"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=100,
                               max_num_windows = 10,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_nw100_w10"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=200,
                               max_num_windows = 10,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_nw200_w10"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=300,
                               max_num_windows = 10,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_nw300_w10"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=300,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_nw300_w20"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=400,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_nw400_w20"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=0,
                               p_threshold = 0.05,
                               max_num_windows = 10,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_p05_w10"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=0,
                               p_threshold = 0.005,
                               max_num_windows = 10,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_p005_w10"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=0,
                               p_threshold = 0.0005,
                               max_num_windows = 10,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_p0005_w10"),
    
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=0,
                               p_threshold = 0.0005,
                               max_num_windows = 10,
                               random_selection = True,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RSFS_p0005_w10"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=200,
                               max_num_windows = 10,
                               random_selection = True,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RSFS_nw200_w10"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=200,
                               random_selection = True,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RSFS_nw200_w20"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=300,
                               max_num_windows = 10,
                               discretization="SFA", 
                               ending_selection=True,
                               random_state=random_state),
        name="ST_SG_ES_nw300_w10"),
    ]

strategies_V1_kws = [ 
    TSCStrategy_proba(
        SearchTechnique_KWS(K=5, discretization="SFA",
                            max_num_windows = 20,
                            func = "mean",
                            random_state=random_state),
        name="ST_KWS_K5_w20_mean"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=10, discretization="SFA",
                            max_num_windows = 20,
                            func = "mean",
                            random_state=random_state),
        name="ST_KWS_K10_w20_mean"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=5, discretization="SFA",
                            max_num_windows = 20,
                            func = "max",
                            random_state=random_state),
        name="ST_KWS_K5_w20_max"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=10, discretization="SFA",
                            max_num_windows = 20,
                            func = "max",
                            random_state=random_state),
        name="ST_KWS_K10_w20_max"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=10, discretization="SFA",
                            max_num_windows = 30,
                            func = "max",
                            random_state=random_state),
        name="ST_KWS_K10_w30_max"),  
    TSCStrategy_proba(
        SearchTechnique_KWS(K=10, method='Declined', discretization="SFA",
                            func = "max",
                            n_words=200,
                            max_num_windows = 20,
                            inclination = 1.3,
                            random_state=random_state),
        name="ST_KWS_K10_w20_max_Declined"), 
    TSCStrategy_proba(
        SearchTechnique_KWS(K=10, discretization="SFA",
                            max_num_windows = 20,
                            func = "max",
                            random_selection=True,
                            random_state=random_state),
        name="ST_KWS_RSFS_K10_w20_max"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=10, method='Declined', discretization="SFA",
                            func = "max",
                            n_words=200,
                            max_num_windows = 20,
                            inclination = 1.3,
                            random_selection=True,
                            random_state=random_state),
        name="ST_KWS_RSFS_K10_w20_max_Declined"),
    ]



# ST_V1_FULL
strategies_V1_FULL = [
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=200,
                               max_num_windows = 10,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_nw200_w10"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=10, discretization="SFA",
                            max_num_windows = 20,
                            func = "max",
                            random_state=random_state),
        name="ST_KWS_K10_w20_max"),
]


# ST_V2
strategies_V2 = [
    TSCStrategy_proba(
        SearchTechnique_MD(random_state=random_state,
                           max_sfa_windows = 10,
                           max_sax_windows = 2,
                           n_sfa_words = 200,
                           n_sax_words = 200),
        name="ST_MD_nw200_200_w10_2"),
    TSCStrategy_proba(
        SearchTechnique_MD(random_state=random_state,
                           max_sfa_windows = 10,
                           max_sax_windows = 4,
                           n_sfa_words = 200,
                           n_sax_words = 200),
        name="ST_MD_nw200_200_w10_4"),
    TSCStrategy_proba(
        SearchTechnique_MD(random_state=random_state,
                           max_sfa_windows = 8,
                           max_sax_windows = 2,
                           n_sfa_words = 200,
                           n_sax_words = 200),
        name="ST_MD_nw200_200_w8_2"),
    TSCStrategy_proba(
        SearchTechnique_MD(random_state=random_state,
                           max_sfa_windows = 12,
                           max_sax_windows = 2,
                           n_sfa_words = 200,
                           n_sax_words = 200),
        name="ST_MD_nw200_200_w12_2"),
    
    
    TSCStrategy_proba(
        SearchTechnique_MD(random_state=random_state,
                           max_sfa_windows = 8,
                           max_sax_windows = 2,
                           random_selection = True,
                           n_sfa_words = 200,
                           n_sax_words = 200),
        name="ST_MD_RSFS_nw200_200_w8_2"),
    TSCStrategy_proba(
        SearchTechnique_MD(random_state=random_state,
                           max_sfa_windows = 8,
                           max_sax_windows = 2,
                           randomize_best_words = True,
                           n_sfa_words = 200,
                           n_sax_words = 200),
        name="ST_MD_FSRS_nw200_200_w8_2"),
    ]


strategies_V2_FULL = [
    TSCStrategy_proba(
        SearchTechnique_MD(random_state=random_state,
                           max_sfa_windows = 8,
                           max_sax_windows = 2,
                           n_sfa_words = 200,
                           n_sax_words = 200),
        name="ST_MD_nw200_200_w8_2"),
    ]


strategies_V3 = [
    
    TSCStrategy_proba(
        SearchTechnique_Ngram(N=5,
                              random_state=random_state),
        name="ST_5grams"),
    TSCStrategy_proba(
        SearchTechnique_Ngram(N=3,
                              random_state=random_state),
        name="ST_3grams"),
    TSCStrategy_proba(
        SearchTechnique_Ngram(N=5, word_length=4,
                              random_state=random_state),
        name="ST_5gram_WL4"),
    TSCStrategy_proba(
        SearchTechnique_Ngram(N=3, word_length=4,
                              random_state=random_state),
        name="ST_3gram_WL4"),
    TSCStrategy_proba(
        SearchTechnique_Ngram(N=5, word_length=4,
                              alphabet_size=2,
                              random_state=random_state),
        name="ST_5gram_WL4_A2"),
    TSCStrategy_proba(
        SearchTechnique_Ngram(N=3, word_length=4,
                              alphabet_size=2,
                              random_state=random_state),
        name="ST_3gram_WL4_A2"),
    ]


strategies_V3_reso = [
                  
    TSCStrategy_proba(
        SearchTechnique_NgramResolution(N=5,
                                        random_state=random_state),
        name="ST_5grams_reso"),
    TSCStrategy_proba(
        SearchTechnique_NgramResolution(N=3,
                                        random_state=random_state),
        name="ST_3grams_reso"),
    TSCStrategy_proba(
        SearchTechnique_NgramResolution(N=5,
                                        n_sfa_words=100, n_sax_words=100,
                                        random_state=random_state),
        name="ST_5grams_reso_nw100_100"),
    TSCStrategy_proba(
        SearchTechnique_NgramResolution(N=3,
                                        n_sfa_words=100, n_sax_words=100,
                                        random_state=random_state),
        name="ST_3grams_reso_nw100_100"),
    TSCStrategy_proba(
        SearchTechnique_NgramResolution(N=5,
                                        n_sfa_words=50, n_sax_words=50,
                                        random_state=random_state),
        name="ST_5grams_reso_nw50_50"),
    TSCStrategy_proba(
        SearchTechnique_NgramResolution(N=3,
                                        n_sfa_words=50, n_sax_words=50,
                                        random_state=random_state),
        name="ST_3grams_reso_nw50_50"),
    TSCStrategy_proba(
        SearchTechnique_NgramResolution(N=3,
                                        n_sfa_words=200, n_sax_words=200,
                                        declined = True,
                                        random_state=random_state),
        name="ST_3grams_reso_nw200_Declined"),
    TSCStrategy_proba(
        SearchTechnique_NgramResolution(N=3,
                                        n_sfa_words=150, n_sax_words=150,
                                        declined = True,
                                        remove_n_words = 30,
                                        random_state=random_state),
        name="ST_3grams_reso_nw150_Declined"),
    TSCStrategy_proba(
        SearchTechnique_NgramResolution(N=3, word_length=4,
                                        random_state=random_state),
        name="ST_3grams_reso_WL4"),
    TSCStrategy_proba(
        SearchTechnique_NgramResolution(N=3, word_length=4,
                                        alphabet_size=2,
                                        random_state=random_state),
        name="ST_3grams_reso_WL4_A2"),
    ]


strategies_V3_FULL = [
    TSCStrategy_proba(
        SearchTechnique_Ngram(N=3, word_length=4,
                              alphabet_size=2,
                              random_state=random_state),
        name="ST_3gram_WL4_A2"),
    TSCStrategy_proba(
        SearchTechnique_NgramResolution(N=3, word_length=4,
                                        alphabet_size=2,
                                        random_state=random_state),
        name="ST_3grams_reso_WL4_A2"), 
    ]


strategies_V4 = [
    TSCStrategy_proba(
        SearchTechnique(random_state=random_state),
        name="ST"),    
    ]

strategy = strategies_V3_FULL
variant = "ST_V3_FULL"

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


result_strategy_path = RESULT_PATH + variant + '/'
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
score_results.to_csv(result_strategy_path+'results.csv')
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




