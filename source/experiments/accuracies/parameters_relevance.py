# -*- coding: utf-8 -*-
import sys
sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace/source')
sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace')

from source.technique import SearchTechnique
from source.experiments.database import get_dataset, get_Xy_from, DATASET_NAMES

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import pandas as pd
import os



pd.set_option("display.width", 500)
pd.set_option("max_colwidth", 80)
pd.set_option("max_columns", 10)


class ParametersRelevance():
    
    folder_path = {
        DATASET_NAMES[0] : 'ecg/',
        DATASET_NAMES[1] : 'worms/'
        }
    
    def execute_all(dataset_name):
        #ParametersRelevance.resolution_relevance(dataset_name)
        #ParametersRelevance.chi2_p_value_relevance(dataset_name)
        ParametersRelevance.chi2_ranking_relevance(dataset_name)
        #ParametersRelevance.word_relevance(dataset_name)
        #ParametersRelevance.classifier_parameters_relevance(dataset_name)
        #ParametersRelevance.resolution_selection_relevance(dataset_name)
    
    def resolution_relevance(dataset_name):
        """
        If you are saving the bag of bags you will save a lot of time but you have
        to select the resolution inside the classifier
        
        """
        
        train, labels = get_Xy_from(dataset_name, split='train')
        ts_length = train.iloc[0,0].size
        
        clf = SearchTechnique(ts_length)
        rm = clf.resolution_matrix.matrix.copy()
        
        for p_threshold in [0.00001, 0.000005, 0.000001]:            
            acc = pd.DataFrame()
            for window in rm:
                for ngram in rm.index:
                    if(rm.loc[ngram,window] > 0):
                        print('window: {}'.format(window))
                        print('ngram: {}'.format(ngram))
                        
                        unique_resolution = pd.DataFrame(data=False, index=rm.index, columns=[window])
                        unique_resolution.loc[ngram,window] = 1            
                        
                        clf = SearchTechnique(ts_length,
                                              word_selection = 'p threshold',
                                              p_threshold = p_threshold,
                                              random_state = 19)
                        clf.resolution_matrix.matrix = unique_resolution   
                        clf.fit(train, labels)
                        
                        
                        test, y_true = get_Xy_from(dataset_name, split='test')
                        y_pred = clf.predict(test)
                        
                        acc.loc[ngram,window] = accuracy_score(y_true, y_pred)     
                        
            acc = acc.fillna('')
            acc.columns.name = 'accuracy'
            print('acc')
            with open(ParametersRelevance.folder_path[dataset_name]+'word_selection.txt', 'a') as f:
                print('keeping all words with p-value less than {} according to chi2'.format(p_threshold),file=f)
                print(acc, file=f)
                print('\n\n', file=f)
    
        for n_words in [70, 80, 100, 110, 120, 200, 300, 400]:        
            acc = pd.DataFrame()
            for window in rm:
                for ngram in rm.index:
                    if(rm.loc[ngram,window] > 0):
                        print('window: {}'.format(window))
                        print('ngram: {}'.format(ngram))
                        
                        unique_resolution = pd.DataFrame(data=False, index=rm.index, columns=[window])
                        unique_resolution.loc[ngram,window] = 1            
                        
                        clf = SearchTechnique(ts_length,
                                              word_selection = 'best n words',
                                              n_words = n_words,
                                              random_state = 19)
                        clf.resolution_matrix.matrix = unique_resolution   
                        clf.fit(train, labels)
                        
                        
                        test, y_true = get_Xy_from(dataset_name, split='test')
                        y_pred = clf.predict(test)
                        
                        acc.loc[ngram,window] = accuracy_score(y_true, y_pred)
                        
            acc = acc.fillna('')
            acc.columns.name = 'accuracy'
            print('acc')
            with open( ParametersRelevance.folder_path[dataset_name]+'word_selection.txt', 'a') as f:
                print('keeping the best {} words according to the chi2 ranking'.format(n_words), file=f)
                print(acc, file=f)
                print('\n\n', file=f)
                
    
    def chi2_p_value_relevance(dataset_name):
        train, labels = get_Xy_from(dataset_name, split='train')
        ts_length = train.iloc[0,0].size
        
        stdout = sys.stdout
        for p_threshold in [0.0005, 0.00005, 0.000005, 0.0000001]:
            print('p_threshold: {}'.format(p_threshold))           
            with open(ParametersRelevance.folder_path[dataset_name]+'word_selection.txt', 'a') as f:
                #sys.stdout = f
                print('p_threshold: {}'.format(p_threshold))         
                        
                clf = SearchTechnique(ts_length,
                                      word_selection = 'p threshold',
                                      p_threshold = p_threshold,
                                      random_state = 19)
                clf.fit(train, labels)        
                        
                test, y_true = get_Xy_from(dataset_name, split='test')
                y_pred = clf.predict(test)
                        
                acc = accuracy_score(y_true, y_pred)
                print('keeping all words with p-value less than {} according to chi2'.format(p_threshold), file=f)
                print('acc = {}'.format(acc))
                print('\n\n')
                sys.stdout = stdout
    
    def chi2_ranking_relevance(dataset_name):
        train, labels = get_Xy_from(dataset_name, split='train')
        ts_length = train.iloc[0,0].size

        verbose=True        
        stdout = sys.stdout
        for n_words in [10, 20, 70, 80, 100, 110, 120, 200, 300, 400, 500, 600, 800, 1000]:
            print('n_words: {}'.format(n_words)) 
            with open(ParametersRelevance.folder_path[dataset_name]+'word_selection.txt', 'a') as f:
                #sys.stdout = f
                print('n_words: {}'.format(n_words))
                        
                clf = SearchTechnique(ts_length,
                                      word_selection = 'best n words',
                                      n_words = n_words,
                                      random_state = 19,
                                      verbose=verbose)
                clf.fit(train, labels)        
                        
                test, y_true = get_Xy_from(dataset_name, split='test')
                y_pred = clf.predict(test)
                        
                acc = accuracy_score(y_true, y_pred)
                #print('keeping the best {} words according to the chi2 ranking'.format(n_words))
                print('acc = {}'.format(acc))
                print('\n\n')
                verbose=False
            sys.stdout = stdout
    
    def word_relevance(dataset_name):
        
        train, labels = get_Xy_from(dataset_name, split='train')
        ts_length = train.iloc[0,0].size
        
        stdout = sys.stdout
        for n_words in [110]:
            print('n_words: {}'.format(n_words)) 
            with open(ParametersRelevance.folder_path[dataset_name]+'best_words.txt', 'a') as f:
                sys.stdout = f
                print('n_words: {}'.format(n_words))         
                        
                clf = SearchTechnique(ts_length,
                                      word_selection = 'best n words',
                                      n_words = n_words,
                                      random_state = 19)
                clf.fit(train, labels)        
                        
                test, y_true = get_Xy_from(dataset_name, split='test')
                y_pred = clf.predict(test)
                        
                acc = accuracy_score(y_true, y_pred)
                print('keeping the best {} words according to the chi2 ranking'.format(n_words))
                print('acc = {}'.format(acc))
                print('selected words:')
                for word_ in clf.selected_words:
                    print(word_)
                    print('\n\n')
                sys.stdout = stdout
    
    
    def classifier_parameters_relevance(dataset_name):
        train, labels = get_Xy_from(dataset_name, split='train')
        ts_length = train.iloc[0,0].size
        
        stdout = sys.stdout
        method = 'highest max'
        print('resolution selection method: {}'.format(method))    
        
        clf_relevance_path = ParametersRelevance.folder_path[dataset_name]+'classifiers_relevance/'
        if(not os.path.exists(clf_relevance_path)):
            os.makedirs(clf_relevance_path)
        
        for n_words in [3,4,5,6,8,10,20,30,40,50,60,70, 80, 100, 110, 120, 200, 300, 400]:
            print('n_words: {}'.format(n_words))
            configs = {
             'default_LR': LogisticRegression(random_state=21),
             'balanced_LR': LogisticRegression(random_state=21,
                                               class_weight = 'balanced'),
             'max_iter_LR': LogisticRegression(random_state=21,
                                               max_iter = 1000),
             'max_iter_balanced_LR': LogisticRegression(random_state=21,
                                               class_weight = 'balanced',
                                               max_iter = 1000),
             'balanced_LRCV': LogisticRegressionCV(random_state=21,
                                                   class_weight='balanced',
                                                   max_iter = 1000),
             'default_SVC': svm.SVC(random_state=21,
                                    class_weight = 'balanced'),
             'balanced_LSVC': svm.LinearSVC(random_state=21,
                                            class_weight='balanced',
                                            dual=False),
             'default_RF': RandomForestClassifier(random_state=21),
             'balanced_RF': RandomForestClassifier(random_state=21,
                                                  class_weight='balanced'),
             'balanced_subsample_RF': RandomForestClassifier(random_state=21,
                                                  class_weight='balanced_subsample'),
            }
            
            for param, clf_ in configs.items():
                print('param: {}'.format(param))
                
                with open(clf_relevance_path+param+'.txt', 'a') as f:
                    sys.stdout = f
                    
                    print('n_words: {}'.format(n_words))                
                    st = SearchTechnique(ts_length,
                                          word_selection = 'best n words',
                                          n_words = n_words,
                                          random_state = 19)
                    st.clf = clf_
                    st.fit(train, labels)        
                            
                    test, y_true = get_Xy_from(dataset_name, split='test')
                    y_pred = st.predict(test)
                            
                    acc = accuracy_score(y_true, y_pred)
                    print('acc = {}'.format(acc))
                    print('')
                    sys.stdout = stdout
    
    def resolution_selection_relevance(dataset_name):    
        
        train, labels = get_Xy_from(dataset_name, split='train')
        ts_length = train.iloc[0,0].size
        
        stdout = sys.stdout
        method = 'highest mean of 10 percents of the highest words'
        print('resolution selection method: {}'.format(method)) 
        with open(ParametersRelevance.folder_path[dataset_name]+'resolution_method.txt', 'a') as f:
            sys.stdout = f
            print('\n\nresolution selection method: {}\n'.format(method))         
            
            for n_words in [70, 80, 100, 110, 120, 200, 300, 400]:
                print('n_words: {}'.format(n_words))
                print('n_words: {}'.format(n_words), file=stdout)
                clf = SearchTechnique(ts_length,
                                      word_selection = 'best n words',
                                      n_words = n_words,
                                      random_state = 19)
                clf.fit(train, labels)        
                        
                test, y_true = get_Xy_from(dataset_name, split='test')
                y_pred = clf.predict(test)
                        
                acc = accuracy_score(y_true, y_pred)
                print('acc = {}'.format(acc))
                print('')
            sys.stdout = stdout

ParametersRelevance.execute_all(DATASET_NAMES[0])