# -*- coding: utf-8 -*-
import sys

# sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace/source')
# sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace')
# sys.path.append('/home/marcio55agb/MasterDegreeWorkspace')
# import required functions and classes
from pathlib import Path
import os
import sktime
import warnings
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import multiprocessing

from source.utils import DatasetHandler, TsHandler
from source.experiments.config import OUTPUT_PATH



from sklearn.metrics import accuracy_score, roc_auc_score  # f1_score, log_loss, balanced_accuracy_score,
from sktime.benchmarking.data import UEADataset, make_datasets
# from sktime.benchmarking.evaluation import Evaluator
# from sktime.benchmarking.metrics import PairwiseMetric, AggregateMetric
from sktime.benchmarking.orchestration import Orchestrator
# from sktime.benchmarking.results import HDDResults
# from sktime.benchmarking.strategies import TSCStrategy, TSCStrategy_proba
from sktime.benchmarking.tasks import TSCTask
from sktime.classification.dictionary_based import (
    ContractableBOSS,
    WEASEL,
    TemporalDictionaryEnsemble
)
# from sktime.classification.shapelet_based import (
#    MrSEQLClassifier
# )
from sktime.series_as_features.model_selection import PresplitFilesCV

# from sktime_changes import PairwiseMetric, AggregateMetric
from source.experiments.sktime_changes import Evaluator, HDDResults, TSCStrategy_proba

from source.utils import draw_cd_diagram, calculate_efficiency
from source.experiments.variant_searching.datasets import DATASET_NAMES, LARGER_DATASETS_NAMES
from source.technique import (
    RandomClassifier,
    SearchTechnique,
    SearchTechniqueCV,
    SearchTechnique_CV_RSFS,
    SearchTechnique_KWS,
    SearchTechnique_SG_CLF,
    SearchTechnique_MD,
    SearchTechnique_Ngram,
    SearchTechnique_NgramResolution,
    SearchTechnique_Ensemble
)


def run_guided_path():
    # warnings.simplefilter(action='ignore', category=UserWarning)

    DatasetHandler.setup_datasets()
    ts_folder_path = TsHandler.get_ts_folder_path()

    # set up paths to data and results folder
    output_variant_search_path = os.path.join(OUTPUT_PATH, "variant_searching/")
    score_path = os.path.join(output_variant_search_path, "scores/")
    result_path = os.path.join(output_variant_search_path, "local_results/")

    # check folders existence
    Path(score_path).mkdir(parents=True, exist_ok=True)
    Path(result_path).mkdir(exist_ok=True)

    datasets = [UEADataset(path=ts_folder_path, name=name) for name in DATASET_NAMES]

    # Alternatively, we can use a helper function to create them automatically
    names = LARGER_DATASETS_NAMES
    names = DATASET_NAMES[:1]

    tasks = [TSCTask(target="target") for _ in range(len(DATASET_NAMES))]

    # Specify learning strategies

    random_state = 27
    strategies = []
    results_paths = []

    def get_strategies():
        strategies_random = [
            TSCStrategy_proba(
                RandomClassifier(random_state=None),
                name="RandomClassifier"
            )
        ]

        strategies_V0 = [
            TSCStrategy_proba(
                SearchTechniqueCV(discretization="SFA", random_state=random_state, ),
                name="ST_CV_SFA"),
            TSCStrategy_proba(
                SearchTechniqueCV(discretization="SFA", random_state=random_state,
                                  max_window_length=1.),
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
                                  max_window_length=1.),
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

        strategies_V0_FINAL = [
            TSCStrategy_proba(
                SearchTechniqueCV(discretization="SFA", feature_selection=True,
                                  n_words=200,
                                  random_state=random_state),
                name="ST_CV_SFA_FS"),
        ]

        strategies_V1_clf = [
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='20',
                                       n_words=100,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_LogisticRegression"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='20',
                                       n_words=100,
                                       random_selection=True,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_RSFS_LogisticRegression"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='21',
                                       n_words=100,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_LogisticRegression_lib"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='01',
                                       n_words=100,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_RandomForest_MF40"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='02',
                                       n_words=100,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_RandomForest"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='02',
                                       n_words=100,
                                       random_selection=True,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_RSFS_RandomForest"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='02',
                                       n_words=100,
                                       ending_selection=True,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_DS_RandomForest"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='03',
                                       n_words=100,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_RandomForest_Entropy"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='04',
                                       n_words=100,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_RandomForest_L2"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='10',
                                       n_words=100,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_SVC_rbf"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='10',
                                       n_words=100,
                                       random_selection=True,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_RSFS_SVC_rbf"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='14',
                                       n_words=100,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_SVC_rbf_balanced"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='11',
                                       n_words=100,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_SVC_poly3"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='12',
                                       n_words=100,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_SVC_poly5"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='13',
                                       n_words=100,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_SVC_sigmoid"),
        ]

        strategies_V1_sg = [
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='02',
                                       n_words=20,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_nw20_w20"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='02',
                                       n_words=50,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_nw50_w20"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='02',
                                       n_words=100,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_nw100_w20"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='02',
                                       n_words=200,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_nw200_w20"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='02',
                                       n_words=20,
                                       max_num_windows=30,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_nw20_w30"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='02',
                                       n_words=50,
                                       max_num_windows=30,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_nw50_w30"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='02',
                                       n_words=100,
                                       max_num_windows=30,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_nw100_w30"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='02',
                                       n_words=200,
                                       max_num_windows=30,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_nw200_w30"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='02',
                                       n_words=50,
                                       max_num_windows=10,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_nw50_w10"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='02',
                                       n_words=100,
                                       max_num_windows=10,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_nw100_w10"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='02',
                                       n_words=200,
                                       max_num_windows=10,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_nw200_w10"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='02',
                                       n_words=300,
                                       max_num_windows=10,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_nw300_w10"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='02',
                                       n_words=300,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_nw300_w20"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='02',
                                       n_words=400,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_nw400_w20"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='02',
                                       n_words=0,
                                       p_threshold=0.05,
                                       max_num_windows=10,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_p05_w10"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='02',
                                       n_words=0,
                                       p_threshold=0.005,
                                       max_num_windows=10,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_p005_w10"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='02',
                                       n_words=0,
                                       p_threshold=0.0005,
                                       max_num_windows=10,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_p0005_w10"),

            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='02',
                                       n_words=0,
                                       p_threshold=0.0005,
                                       max_num_windows=10,
                                       random_selection=True,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_RSFS_p0005_w10"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='02',
                                       n_words=200,
                                       max_num_windows=10,
                                       random_selection=True,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_RSFS_nw200_w10"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='02',
                                       n_words=200,
                                       random_selection=True,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_RSFS_nw200_w20"),
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='02',
                                       n_words=300,
                                       max_num_windows=10,
                                       discretization="SFA",
                                       ending_selection=True,
                                       random_state=random_state),
                name="ST_SG_ES_nw300_w10"),
        ]

        strategies_V1_kws = [
            TSCStrategy_proba(
                SearchTechnique_KWS(K=5, discretization="SFA",
                                    max_num_windows=20,
                                    func="mean",
                                    random_state=random_state),
                name="ST_KWS_K5_w20_mean"),
            TSCStrategy_proba(
                SearchTechnique_KWS(K=10, discretization="SFA",
                                    max_num_windows=20,
                                    func="mean",
                                    random_state=random_state),
                name="ST_KWS_K10_w20_mean"),
            TSCStrategy_proba(
                SearchTechnique_KWS(K=5, discretization="SFA",
                                    max_num_windows=20,
                                    func="max",
                                    random_state=random_state),
                name="ST_KWS_K5_w20_max"),
            TSCStrategy_proba(
                SearchTechnique_KWS(K=10, discretization="SFA",
                                    max_num_windows=20,
                                    func="max",
                                    random_state=random_state),
                name="ST_KWS_K10_w20_max"),
            TSCStrategy_proba(
                SearchTechnique_KWS(K=10, discretization="SFA",
                                    max_num_windows=30,
                                    func="max",
                                    random_state=random_state),
                name="ST_KWS_K10_w30_max"),
            TSCStrategy_proba(
                SearchTechnique_KWS(K=10, method='Declined', discretization="SFA",
                                    func="max",
                                    n_words=200,
                                    max_num_windows=20,
                                    inclination=1.3,
                                    random_state=random_state),
                name="ST_KWS_K10_w20_max_Declined"),
            TSCStrategy_proba(
                SearchTechnique_KWS(K=10, discretization="SFA",
                                    max_num_windows=20,
                                    func="max",
                                    random_selection=True,
                                    random_state=random_state),
                name="ST_KWS_RSFS_K10_w20_max"),
            TSCStrategy_proba(
                SearchTechnique_KWS(K=10, method='Declined', discretization="SFA",
                                    func="max",
                                    n_words=200,
                                    max_num_windows=20,
                                    inclination=1.3,
                                    random_selection=True,
                                    random_state=random_state),
                name="ST_KWS_RSFS_K10_w20_max_Declined"),
        ]

        # ST_V1_FULL
        strategies_V1_FINAL = [
            TSCStrategy_proba(
                SearchTechnique_SG_CLF(clf_name='02',
                                       n_words=200,
                                       max_num_windows=10,
                                       discretization="SFA",
                                       random_state=random_state),
                name="ST_SG_nw200_w10"),
            TSCStrategy_proba(
                SearchTechnique_KWS(K=10, discretization="SFA",
                                    max_num_windows=20,
                                    func="max",
                                    random_state=random_state),
                name="ST_KWS_K10_w20_max"),
        ]

        # ST_V2
        strategies_V2 = [
            TSCStrategy_proba(
                SearchTechnique_MD(random_state=random_state,
                                   max_sfa_windows=10,
                                   max_sax_windows=2,
                                   n_sfa_words=200,
                                   n_sax_words=200),
                name="ST_MD_nw200_200_w10_2"),
            TSCStrategy_proba(
                SearchTechnique_MD(random_state=random_state,
                                   max_sfa_windows=10,
                                   max_sax_windows=4,
                                   n_sfa_words=200,
                                   n_sax_words=200),
                name="ST_MD_nw200_200_w10_4"),
            TSCStrategy_proba(
                SearchTechnique_MD(random_state=random_state,
                                   max_sfa_windows=8,
                                   max_sax_windows=2,
                                   n_sfa_words=200,
                                   n_sax_words=200),
                name="ST_MD_nw200_200_w8_2"),
            TSCStrategy_proba(
                SearchTechnique_MD(random_state=random_state,
                                   max_sfa_windows=12,
                                   max_sax_windows=2,
                                   n_sfa_words=200,
                                   n_sax_words=200),
                name="ST_MD_nw200_200_w12_2"),

            TSCStrategy_proba(
                SearchTechnique_MD(random_state=random_state,
                                   max_sfa_windows=8,
                                   max_sax_windows=2,
                                   random_selection=True,
                                   n_sfa_words=200,
                                   n_sax_words=200),
                name="ST_MD_RSFS_nw200_200_w8_2"),
            TSCStrategy_proba(
                SearchTechnique_MD(random_state=random_state,
                                   max_sfa_windows=8,
                                   max_sax_windows=2,
                                   randomize_best_words=True,
                                   n_sfa_words=200,
                                   n_sax_words=200),
                name="ST_MD_FSRS_nw200_200_w8_2"),
        ]

        strategies_V2_FINAL = [
            TSCStrategy_proba(
                SearchTechnique_MD(random_state=random_state,
                                   max_sfa_windows=8,
                                   max_sax_windows=2,
                                   n_sfa_words=200,
                                   n_sax_words=200),
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
                SearchTechnique_Ngram(N=5,
                                      random_selection=True,
                                      random_state=random_state),
                name="ST_RSFS_5grams"),
            TSCStrategy_proba(
                SearchTechnique_Ngram(N=3,
                                      random_selection=True,
                                      random_state=random_state),
                name="ST_RSFS_3grams"),
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
                                                declined=True,
                                                random_state=random_state),
                name="ST_3grams_reso_nw200_Declined"),
            TSCStrategy_proba(
                SearchTechnique_NgramResolution(N=3,
                                                n_sfa_words=150, n_sax_words=150,
                                                declined=True,
                                                remove_n_words=30,
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

        strategies_V3_FINAL = [
            TSCStrategy_proba(
                SearchTechnique_Ngram(N=3, word_length=4,
                                      alphabet_size=2,
                                      random_state=random_state),
                name="ST_3gram_WL4_A2"),
            TSCStrategy_proba(
                SearchTechnique_NgramResolution(N=3, word_length=4,
                                                random_state=random_state),
                name="ST_3grams_reso_WL4"),
        ]

        strategies_V4 = [
            TSCStrategy_proba(
                SearchTechnique_Ensemble(num_clfs=10,
                                         random_state=random_state,
                                         method=1,
                                         sfa_per_sax=3,
                                         data_frac=.2),
                name="Ensemble_S10"),
            TSCStrategy_proba(
                SearchTechnique_Ensemble(num_clfs=10,
                                         random_state=random_state,
                                         sfa_window_per_slc=4,
                                         sax_window_per_slc=1,
                                         method=1,
                                         sfa_per_sax=3,
                                         data_frac=.2),
                name="Ensemble_S10_halfW"),
            TSCStrategy_proba(
                SearchTechnique_Ensemble(num_clfs=60,
                                         random_state=random_state,
                                         method=2,
                                         sfa_per_sax=10,
                                         data_frac=.05),
                name="Ensemble_S60_singleW"),
            TSCStrategy_proba(
                SearchTechnique_Ensemble(num_clfs=40,
                                         random_state=random_state,
                                         method=2,
                                         sfa_per_sax=10,
                                         data_frac=.05),
                name="Ensemble_S40_singleW"),
            TSCStrategy_proba(
                SearchTechnique_Ensemble(num_clfs=60,
                                         random_state=random_state,
                                         method=2,
                                         sfa_per_sax=10,
                                         data_frac=.1),
                name="Ensemble_S60_singleW_frac01"),
            TSCStrategy_proba(
                SearchTechnique_Ensemble(num_clfs=40,
                                         random_state=random_state,
                                         method=2,
                                         sfa_per_sax=10,
                                         data_frac=.1),
                name="Ensemble_S40_singleW_frac01"),
            TSCStrategy_proba(
                SearchTechnique_Ensemble(num_clfs=40,
                                         random_state=random_state,
                                         method=2,
                                         sfa_per_sax=10,
                                         data_frac=.2),
                name="Ensemble_S40_singleW_frac02"),
            TSCStrategy_proba(
                SearchTechnique_Ensemble(num_clfs=20,
                                         random_state=random_state,
                                         sfa_window_per_slc=4,
                                         sax_window_per_slc=1,
                                         method=1,
                                         sfa_per_sax=3,
                                         data_frac=.2),
                name="Ensemble_S20_halfW"),
            TSCStrategy_proba(
                SearchTechnique_Ensemble(num_clfs=30,
                                         random_state=random_state,
                                         method=2,
                                         sfa_per_sax=10,
                                         data_frac=.2),
                name="Ensemble_S30_singleW_frac02"),
            TSCStrategy_proba(
                SearchTechnique_Ensemble(num_clfs=30,
                                         random_state=random_state,
                                         method=2,
                                         sfa_per_sax=10,
                                         data_frac=.3),
                name="Ensemble_S30_singleW_frac03"),
            TSCStrategy_proba(
                SearchTechnique_Ensemble(num_clfs=10,
                                         random_state=random_state,
                                         sfa_window_per_slc=4,
                                         sax_window_per_slc=1,
                                         method=1,
                                         sfa_per_sax=3,
                                         data_frac=.3),
                name="Ensemble_S10_halfW_frac03"),
            TSCStrategy_proba(
                SearchTechnique_Ensemble(num_clfs=10,
                                         random_state=random_state,
                                         sfa_window_per_slc=4,
                                         sax_window_per_slc=1,
                                         method=1,
                                         sfa_per_sax=3,
                                         data_frac=.6),
                name="Ensemble_S10_halfW_frac06"),
            TSCStrategy_proba(
                SearchTechnique_Ensemble(num_clfs=30,
                                         random_state=random_state,
                                         method=2,
                                         sfa_per_sax=10,
                                         data_frac=.6),
                name="Ensemble_S30_singleW_frac06"),
            TSCStrategy_proba(
                SearchTechnique_Ensemble(num_clfs=30,
                                         random_state=random_state,
                                         method=2,
                                         sfa_per_sax=5,
                                         data_frac=.6),
                name="Ensemble_S30_singleW_frac06_sfa5"),
            TSCStrategy_proba(
                SearchTechnique_Ensemble(num_clfs=30,
                                         random_state=random_state,
                                         method=2,
                                         sfa_per_sax=3,
                                         n_jobs=5,
                                         data_frac=.6),
                name="Ensemble_S30_singleW_frac06_sfa3_parallel5"),
            TSCStrategy_proba(
                SearchTechnique_Ensemble(num_clfs=10,
                                         random_state=random_state,
                                         sfa_window_per_slc=4,
                                         sax_window_per_slc=1,
                                         method=1,
                                         sfa_per_sax=3,
                                         data_frac=.6),
                name="Ensemble_S10_halfW_frac06_parallel3"),
            TSCStrategy_proba(
                SearchTechnique_Ensemble(num_clfs=30,
                                         random_state=random_state,
                                         method=2,
                                         sfa_per_sax=3,
                                         n_jobs=5,
                                         data_frac=.4),
                name="Ensemble_S30_singleW_frac04_sfa3_parallel5"),
            TSCStrategy_proba(
                SearchTechnique_Ensemble(num_clfs=10,
                                         random_state=random_state,
                                         sfa_window_per_slc=4,
                                         sax_window_per_slc=1,
                                         method=1,
                                         sfa_per_sax=3,
                                         data_frac=.5),
                name="Ensemble_S10_halfW_frac05_parallel3"),
            TSCStrategy_proba(
                SearchTechnique_Ensemble(num_clfs=30,
                                         random_state=random_state,
                                         method=2,
                                         sfa_per_sax=4,
                                         n_jobs=5,
                                         data_frac=.6),
                name="Ensemble_S30_singleW_frac02_sfa4_parallel6"),
        ]

        strategies_V4_FINAL = [
            TSCStrategy_proba(
                SearchTechnique_Ensemble(num_clfs=10,
                                         random_state=random_state,
                                         sfa_window_per_slc=4,
                                         sax_window_per_slc=1,
                                         method=1,
                                         sfa_per_sax=3,
                                         data_frac=.5),
                name="Ensemble_S10_halfW_frac05_parallel3"),
        ]

        # strategies_mrseql = [
        #    TSCStrategy_proba(
        #        MrSEQLClassifier(symrep=('sax', 'sfa')),
        #        name="MrSEQL")
        # ]

        strategies_dict = {
            "ST_random": strategies_random,
            "ST_V0": strategies_V0,
            "ST_V0_FINAL": strategies_V0_FINAL,
            "ST_V1_clf": strategies_V1_clf,
            "ST_V1_sg": strategies_V1_sg,
            "ST_V1_kws": strategies_V1_kws,
            "ST_V1_FINAL": strategies_V1_FINAL,
            "ST_V2": strategies_V2,
            "ST_V2_FINAL": strategies_V2_FINAL,
            "ST_V3": strategies_V3,
            "ST_V3_reso": strategies_V3_reso,
            "ST_V3_FINAL": strategies_V3_FINAL,
            "ST_V4": strategies_V4,
            "ST_V4_FINAL": strategies_V4_FINAL
        }
        return strategies_dict

    random_clf_acc = 40.5545
    random_clf_auc = 49.5794
    strategies = get_strategies()
    print(strategies.items())
    for variant, strategy in strategies.items():
        print(f'Running experiments with variant {variant}\n\n')

        score_strategy_path = score_path + variant

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
        auc_scores = evaluator.get_all_datasets_scores('AUC', roc_auc_score, probabilties=True, labels=True,
                                                       multi_class='ovr')

        print('Results:')
        print(f'accuracy: {acc_scores}')
        print(f'auc: {auc_scores}')

        result_strategy_path = result_path + variant + '/'
        if not os.path.isdir(result_strategy_path):
            os.mkdir(result_strategy_path)

        # draw_cd_diagram(df_perf=acc_scores, image_path=result_path, title='Accuracy', labels=True)

        acc_scores = acc_scores.groupby('strategy_name').mean() * 100
        auc_scores = auc_scores.groupby('strategy_name').mean() * 100

        score_results = pd.concat([acc_scores, auc_scores, runtime], axis=1)
        score_results.columns = ['Accuracy mean', 'AUC mean', 'fit runtime mean', 'predict runtime mean']
        print(score_results.iloc[:, 1:])

        score_results['Accuracy efficiency'] = calculate_efficiency(score_results.iloc[:, [0, 2, 3]],
                                                                    random_clf_acc)
        score_results['AUC efficiency'] = calculate_efficiency(score_results.iloc[:, [1, 2, 3]],
                                                               random_clf_auc)
        score_results = score_results.sort_values('Accuracy mean')
        score_results = score_results.round(3)
        score_results.to_csv(result_strategy_path + 'results.csv')
        print(score_results.iloc[:, :2])
        score_results = score_results.sort_values('AUC efficiency')
        print(score_results.iloc[:, -2:])

        fig, ax = plt.subplots(figsize=[8, 6], dpi=200)

        n_strategy = score_results.shape[0]
        colors = np.arange(n_strategy)
        scatter = ax.scatter(score_results['fit runtime mean'],
                             score_results['Accuracy mean'],
                             label=score_results.index.values,
                             c=colors, cmap='viridis')

        handles, labels = scatter.legend_elements(prop="colors")
        legend1 = ax.legend(handles, score_results.index.values,
                            loc="center right", title="Classes",
                            fancybox=True,
                            title_fontsize=14,
                            borderpad=1.0)

        ax.set_title('Accuracy efficiency')
        ax.set_xlabel('fit runtime mean')
        ax.set_ylabel('accuracy mean')
        ax.add_artist(legend1)
        ax.grid(True)

        fig.show()
        plt.savefig(result_strategy_path + 'acc_efficiency')

        fig, ax = plt.subplots(figsize=[8, 6], dpi=200)

        n_strategy = score_results.shape[0]
        colors = np.arange(n_strategy)
        scatter = ax.scatter(score_results['fit runtime mean'],
                             score_results['AUC mean'],
                             label=score_results.index.values,
                             c=colors, cmap='viridis')

        handles, labels = scatter.legend_elements(prop="colors")
        legend1 = ax.legend(handles, score_results.index.values,
                            loc="center right", title="Classes",
                            fancybox=True,
                            title_fontsize=14,
                            borderpad=1.0)

        ax.set_title('AUC efficiency')
        ax.set_xlabel('fit runtime mean')
        ax.set_ylabel('AUC mean')
        ax.add_artist(legend1)
        ax.grid(True)

        fig.show()
        plt.savefig(result_strategy_path + 'auc_efficiency')
        break
