import os
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from source.classifiers import Classifier3M
from source.utils import DatasetHandler


def run_3m_tuning_process(random_state=3):
    print('Running tuning process\n\n')
    distributions = dict(ngram=np.arange(1, 6),
                         word_length=np.arange(3, 17),
                         alphabet_size=np.arange(2, 6),
                         max_window_length=np.linspace(.3, .7, num=30),
                         n_sfa_resolutions=np.arange(2, 13),
                         n_sax_resolutions=np.arange(2, 13),
                         sfa_features_percentile=np.arange(0, 101, 5),  # quantity of words to describe the series
                         # (best n words according to chi2 rank)
                         sax_features_percentile=np.arange(0, 101, 5),  # quantity of words to describe the series
                         # (best n words according to chi2 rank)
                         random_selection=np.array([True, False]),
                         # randomly choose half of extracted words to describe the serie
                         normalize=np.array([True, False]),  # window normalization
                         )

    for size_type, df_names in zip(['widest', 'longest'], [DatasetHandler.get_widest_datasets(),
                                                           DatasetHandler.get_longest_datasets()]):
        for df_name in df_names:
            df_name = 'Crop'
            print(df_name)
            score_path = f'results/experiments/hyperparam_tuning/scores/{size_type}_{df_name}.csv'
            a = 0
            if os.path.exists(score_path):
                a = pd.read_csv(score_path).shape[0]

            train_x, train_y, _, _ = DatasetHandler.get_split_data(df_name)
            for seed in range(a, 10):
                seed = 0
                print(seed, datetime.datetime.now().strftime("%H:%M:%S"))
                cv = StratifiedKFold(4, shuffle=True, random_state=seed)
                clf_3m = Classifier3M(random_state=seed, verbose=False)
                clf = RandomizedSearchCV(clf_3m, distributions, n_iter=1, scoring='accuracy', cv=cv,
                                         random_state=seed, verbose=True, n_jobs=4)
                results = clf.fit(train_x, train_y)
                print('Done')
                scores = pd.DataFrame.from_dict(results.cv_results_)
                if seed == 0:
                    scores.to_csv(score_path, mode='a', index=False)
                else:
                    scores.to_csv(score_path, mode='a', index=False, header=False)
                exit(1)