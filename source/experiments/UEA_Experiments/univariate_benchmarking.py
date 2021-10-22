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

from sklearn.metrics import accuracy_score, roc_auc_score#f1_score, log_loss, balanced_accuracy_score, 
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

from source.utils import draw_cd_diagram, calculate_efficiency
from datasets.univariate.config import DATASET_NAMES, LARGER_DATASETS_NAMES
from source.technique import (
    RandomClassifier,
    SearchTechnique,
    SearchTechniqueCV,
    SearchTechnique_CV_RFSF,
    SearchTechnique_KWS,
    SearchTechnique_SG_CLF,
    SearchTechnique_MR,
    SearchTechnique_Ngram
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

RANDOM_CLF_ACC_mean = .405545
RANDOM_CLF_ROC_AUC_mean = .495794

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

strategies_V1_clf = [
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '20',
                               n_words=100,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_LogisticRegression"),
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
                               n_words=200,
                               random_selection=True,
                               rand_words=10,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_RandomForest_rand10"),
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

strategies_V1_windows_words = [
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               max_num_windows=40,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_w40"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               max_num_windows=50,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_w50"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               max_num_windows=100,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_w100"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=5,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_nw5"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=10,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_nw10"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=20,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_nw20"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=20,
                               max_num_windows=50,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_nw20_w50"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=10,
                               max_num_windows=50,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_nw10_w50"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=5,
                               max_num_windows=50,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_nw5_w50"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=50,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG_nw50_w20"),    
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=None,
                               total_n_words = 200,
                               random_state=random_state),
        name="ST_SG_tw200_w20"),   
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               max_num_windows = 40,
                               n_words=None,
                               total_n_words = 200,
                               random_state=random_state),
        name="ST_SG_tw200_w40"),  
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               max_num_windows = 80,
                               n_words=None,
                               total_n_words = 200,
                               random_state=random_state),
        name="ST_SG_tw200_w80"),  
    ]

strategies_V1 = [
    TSCStrategy_proba(
        SearchTechnique_KWS(K=1, method='Equal', discretization="SFA",
                            random_state=random_state),
        name="ST_KWS_K1_Equal"), 
    TSCStrategy_proba(
        SearchTechnique_KWS(K=2, method='Equal', discretization="SFA",
                            random_state=random_state),
        name="ST_KWS_K2_Equal"), 
    TSCStrategy_proba(
        SearchTechnique_KWS(K=5, method='Equal', discretization="SFA",
                            random_state=random_state),
        name="ST_KWS_K5_Equal"), 
    TSCStrategy_proba(
        SearchTechnique_KWS(K=10, method='Equal', discretization="SFA",
                            random_state=random_state),
        name="ST_KWS_K10_Equal"), 
    TSCStrategy_proba(
        SearchTechnique_KWS(K=20, method='Equal', discretization="SFA",
                            random_state=random_state),
        name="ST_KWS_K20_Equal"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=10, method='Declined', discretization="SFA",
                            n_words=60,
                            inclination = 1.3,
                            random_state=random_state),
        name="ST_KWS_K10_Declined"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=5, method='Equal', discretization="SFA",
                            random_top_words=True,
                            random_state=random_state),
        name="ST_KWS_K5_Equal_RS"), 
    TSCStrategy_proba(
        SearchTechnique_KWS(K=10, method='Declined', discretization="SFA",
                            random_top_words=True,
                            n_words=60,
                            inclination = 1.3,
                            random_state=random_state),
        name="ST_KWS_K10_Declined_RS"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=20, method='Declined', discretization="SFA",
                            n_words=40,
                            inclination = 1.15,
                            random_state=random_state),
        name="ST_KWS_K20_Declined"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=10,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=10,
                               discretization="SFA", 
                               ending_selection=True,
                               random_state=random_state),
        name="ST_SG_EndingSelection"),
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=10,
                               discretization="SFA", 
                               random_selection=True,
                               random_state=random_state),
        name="ST_SG_RandomSelection"),
    ]



# ST_V1_FULL
strategies_V1_FULL = [
    TSCStrategy_proba(
        SearchTechnique_SG_CLF(clf_name = '02',
                               n_words=10,
                               discretization="SFA", 
                               random_state=random_state),
        name="ST_SG"),
    TSCStrategy_proba(
        SearchTechnique_KWS(K=10, method='Declined', discretization="SFA",
                            n_words=60,
                            inclination = 1.3,
                            random_state=random_state),
        name="ST_KWS_K10_Declined"),
]


# ST_V2
strategies_V2 = [
    TSCStrategy_proba(
        SearchTechnique_MR(random_state=random_state),
        name="ST_MR"),
    TSCStrategy_proba(
        SearchTechnique_MR(random_state=random_state,
                           fixed_words=True),
        name="ST_MR_nw10"),
    TSCStrategy_proba(
        SearchTechnique_MR(random_state=random_state,
                           fixed_words=True,
                           max_sax_windows = 4),
        name="ST_MR_nw10_20_4"),
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

strategies_V2_FULL = [
    TSCStrategy_proba(
        SearchTechnique_MR(random_state=random_state,
                           fixed_words=True),
        name="ST_MR_nw10"),
    ]

strategies_V3 = [
    TSCStrategy_proba(
        SearchTechnique_Ngram(random_state=random_state),
        name="ST_Ngram"),
    ]

strategy = strategies_V3
result_path = RESULTS_PATH + "ST_V3/"


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

acc_scores = evaluator.get_all_datasets_scores('Accuracy', accuracy_score)
roc_scores = evaluator.get_all_datasets_scores('ROC AUC', roc_auc_score, probabilties=True, labels=True, multi_class='ovr')
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
score_results = score_results.sort_values('Accuracy mean')
score_results = score_results.round(3)
score_results.to_csv(result_path+'scores.csv')
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

plt.savefig(result_path+'acc_eficiency')


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

plt.savefig(result_path+'roc_eficiency')




