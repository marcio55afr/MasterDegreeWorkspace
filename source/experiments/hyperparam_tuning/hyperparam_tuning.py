import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, StratifiedKFold, ParameterSampler

from source.classifiers import Classifier3M
from source.utils import DatasetHandler

SCORE_PATH = 'results/hyperparam_tuning/3m_scores'

def run_3m_tuning_process(seed=55, n_params=100, n_folds=4):
    rng = np.random.RandomState(seed)

    print('Running tuning process\n\n')
    param_distributions = dict(ngram=np.arange(1, 6),
                               word_length=np.arange(3, 17),
                               alphabet_size=np.arange(2, 6),
                               max_window_length=np.linspace(.3, .7, num=30),
                               n_sax_resolutions=np.arange(2, 7),
                               rate_sfa_resolutions=np.arange(1, 5.1, .2),
                               sfa_features_percentile=np.arange(1, 101, 5),  # quantity of words to describe the series
                               # (best n words according to chi2 rank)
                               sax_features_percentile=np.arange(1, 101, 5),  # quantity of words to describe the series
                               # (best n words according to chi2 rank)
                               random_selection=np.array([True, False]),
                               # randomly choose half of extracted words to describe the serie
                               normalize=np.array([True, False]),  # window normalization
                               )

    param_list = list(ParameterSampler(param_distributions=param_distributions,
                                       n_iter=n_params, random_state=rng))

    size_type = 'medium'
    for df_name in DatasetHandler.get_medium_datasets():
        score_path = f'{SCORE_PATH}/{size_type}/{df_name}.csv'
        a = 0
        if os.path.exists(score_path):
            a = pd.read_csv(score_path).shape[0]

        train_x, train_y, _, _ = DatasetHandler.get_split_data(df_name)
        for i in range(a, n_params):
            print(df_name)
            print(i, datetime.datetime.now().strftime("%H:%M:%S"))

            params = param_list[i]

            folds = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
            clf_3m = Classifier3M(**params, random_state=seed, verbose=False)
            metrics = ['accuracy', 'balanced_accuracy', 'f1_micro', 'f1_weighted', 'neg_log_loss',
                       'roc_auc_ovr', 'roc_auc_ovr_weighted', 'roc_auc_ovo']
            results = cross_validate(clf_3m, train_x, train_y, scoring=metrics, cv=folds,
                                     n_jobs=n_folds, verbose=10, error_score='raise')
            print('Done')

            scores = pd.DataFrame.from_dict(results)
            print(scores)
            scores = pd.DataFrame(scores.mean()).T
            print(scores)
            if i == 0:
                scores.to_csv(score_path, mode='w', index=False)
            else:
                scores.to_csv(score_path, mode='a', index=False, header=False)


def run_3m_tuning_size_comparison(size_type, dataset_names):
    for df_name in dataset_names:
        score_path = f'{SCORE_PATH}/{size_type}/{df_name}.csv'
        df = pd.read_csv(score_path, index_col=None)
        plt.scatter(x=df.fit_time, y=df.test_accuracy)
        plt.show()


def run_3m_tuning_comparison():
    size = 'medium'
    ds_names = DatasetHandler.get_medium_datasets()
    run_3m_tuning_size_comparison(size, ds_names)

    size = 'longest'
    ds_names = DatasetHandler.get_longest_datasets()
    run_3m_tuning_size_comparison(size, ds_names)

    size = 'widest'
    ds_names = DatasetHandler.get_widest_datasets()
    run_3m_tuning_size_comparison(size, ds_names)
