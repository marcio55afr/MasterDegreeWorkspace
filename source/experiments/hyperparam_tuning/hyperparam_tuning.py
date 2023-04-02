import os
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold, ParameterSampler

from source.classifiers import Classifier3M
from source.utils import DatasetHandler


def run_3m_tuning_process(seed=55, n_params=100, n_folds=4):
    rng = np.random.RandomState(seed)

    print('Running tuning process\n\n')
    param_distributions = dict(ngram=np.arange(1, 6),
                               word_length=np.arange(3, 17),
                               alphabet_size=np.arange(2, 6),
                               max_window_length=np.linspace(.3, .7, num=30),
                               n_sfa_resolutions=np.arange(2, 13),
                               n_sax_resolutions=np.arange(2, 13),
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

    for size_type, df_names in zip(['widest', 'longest'], [DatasetHandler.get_widest_datasets(),
                                                           DatasetHandler.get_longest_datasets()]):
        for df_name in df_names:
            df_name = 'SmoothSubspace'
            print(df_name)
            score_path = f'results/hyperparam_tuning/3m_scores/{size_type}_{df_name}.csv'
            a = 0
            if os.path.exists(score_path):
                a = pd.read_csv(score_path).shape[0]

            train_x, train_y, _, _ = DatasetHandler.get_split_data(df_name)
            for i in range(a, n_params):
                i=2
                print(i, datetime.datetime.now().strftime("%H:%M:%S"))

                params = param_list[i]

                folds = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
                clf_3m = Classifier3M(**params, random_state=seed, verbose=False)
                results = cross_validate(clf_3m, train_x, train_y, scoring=['accuracy', 'roc_auc_ovo'], cv=folds,# n_jobs=n_folds,
                                         verbose=3)
                print('Done')

                scores = pd.DataFrame.from_dict(results)
                print(scores)
                scores = pd.DataFrame(scores.mean()).T
                print(scores)
                if i == 0:
                    scores.to_csv(score_path, mode='w', index=False)
                else:
                    scores.to_csv(score_path, mode='a', index=False, header=False)
                input('waiting to continue..')
