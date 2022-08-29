# -*- coding: utf-8 -*-

import sys
sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace')
from source.utils import calculate_efficiency
from source.experiments.UEA_Experiments.datasets.config import LARGER_DATASETS_NAMES

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_benchmark_summary():

    time_path = "official_UEA_results\\MegaComparison\\TimingsRAW\\TRAIN\\TRAINFOLDTrainTimesS\\"
    benchmark_strategies = os.listdir(time_path)
    
    times = pd.Series(dtype=np.float64)
    for clf in benchmark_strategies:
        file = time_path + clf
        df = pd.read_csv(file, index_col=0)
        time = df.loc[LARGER_DATASETS_NAMES].mean(axis=1).mean()
        
        clf_name = clf.split('_')[0]
        times.loc[clf_name] = time
    
    
    auroc_path = "official_UEA_results\\MegaComparison\\AUROC\\TEST\\TESTFOLDAUROCS\\"
    benchmark_strategies = os.listdir(auroc_path)
        
    aurocs = pd.Series(dtype=np.float32)
    for clf in benchmark_strategies:
        file = auroc_path + clf
        df = pd.read_csv(file, index_col=0)
        score = df.loc[LARGER_DATASETS_NAMES].mean(axis=1).mean()
        
        clf_name = clf.split('_')[0]
        aurocs.loc[clf_name] = score
    
    
    acc_path = "official_UEA_results\\MegaComparison\\ACC\\TEST\\TESTFOLDACCS\\"
    benchmark_strategies = os.listdir(acc_path)
        
    accs = pd.Series(dtype=np.float32)
    for clf in benchmark_strategies:
        file = acc_path + clf
        df = pd.read_csv(file, index_col=0)
        score = df.loc[LARGER_DATASETS_NAMES].mean(axis=1).mean()
        
        clf_name = clf.split('_')[0]
        accs.loc[clf_name] = score
    
    df = pd.DataFrame(index=aurocs.index)
    df.index.name = 'strategy'
    df['auroc'] = aurocs
    df['accuracy'] = accs
    df['runtime'] = times
    df.to_csv('benchmark_summary.csv')



def plot_auroc_efficiency():

    summary_file = "benchmark_summary.csv"
    if not os.path.isfile(summary_file):
        extract_benchmark_summary()
    df = pd.read_csv(summary_file, index_col=0)
    
    #file = 'results_windows/ST_MrSEQL_FULL/results.csv'    
    #mr_seql = pd.read_csv(file)
    #mr_seql = mr_seql.set_index('strategy_name')
    #mr_seql = mr_seql.iloc[:,[1,0,2]]
    #mr_seql.iloc[:,:2] = mr_seql.iloc[:,:2]/100
    #mr_seql.columns = df.columns
    #df = pd.concat([df, mr_seql])
    
    
    avg_train_runtimes = df['runtime']
    avg_auroc = df['auroc']*100
    classifiers = avg_train_runtimes.index.values
    
    if (avg_train_runtimes.index.values != avg_auroc.index.values).any():
        raise RuntimeError("The results doesnt match")
    
    
    fig, ax = plt.subplots(figsize=[8,6], dpi=200)
    
    colors = np.arange(avg_auroc.size)
    scatter = ax.scatter(avg_train_runtimes,
                         avg_auroc,
                         label=classifiers,
                         s = 70,
                         c=colors, cmap='viridis')
    
    handles, labels = scatter.legend_elements(prop="colors")
    #legend1 = ax.legend(handles, avg_train_runtimes.index.values ,
    #                    loc="lower right", title="Classes",
    #                    fancybox=True,
    #                    title_fontsize=14,
    #                    borderpad=1.0)
    
    
    for i, name in enumerate(classifiers):
        
        x = avg_train_runtimes[i]*1.22
        y = avg_auroc[i]
        
        bar_len = (x, x*4)
        if name in ['HIVE-COTE v1.0','ProximityForest']:
            bar_len = (x, x*10)
        elif name in ['RISE', 'ResNet']:
            bar_len = (x, x*3)
        elif name in ['STC', 'TSF']:
            bar_len = (x, x*2)
        
        if name == 'InceptionTime':
            y -= .35
            ax.annotate(name, (x,y), fontsize='small')
            y -= 0.02
            ax.plot((x, x*8), (y,y), linewidth=.5, color='black')
            ax.plot((avg_train_runtimes[i], x), (avg_auroc[i],y), linewidth=.5, color='black')
        elif name in ['Catch22','BOSS']:
            y -= .25
            ax.annotate(name, (x,y), fontsize='small')
            y -= 0.02
            ax.plot((x, x*3), (y,y), linewidth=.5, color='black')
            ax.plot((avg_train_runtimes[i], x), (avg_auroc[i],y), linewidth=.5, color='black')
        else:
            y += .15
            ax.annotate(name, (x,y), fontsize='small')
            y -= 0.02
            ax.plot(bar_len, (y,y), linewidth=.5, color='black')
            ax.plot((avg_train_runtimes[i], x), (avg_auroc[i],y), linewidth=.5, color='black')
    
    ax.set_title('Eficiência - AUROC')
    ax.set_xlabel('média do tempo de treinamento (s)')
    ax.set_ylabel('média da área sobre a curva ROC (%)')
    #ax.set_title('AUROC Efficiency')
    #ax.set_xlabel('Average fitting time (s)')
    #ax.set_ylabel('Average area under the ROC curve (%)')
    ax.set_xscale('log')
    ax.set_xlim(1100, 11**8)
    ax.grid(True, alpha=.2)
    ax.set_axisbelow(True)
    
    plt.savefig('benchmark_auroc')
    
    

    def exponential(x):
        return 2**(x/4)
    
    ax.set_xlim(20, 12**8)
    base_value = 49.5794
    auroc_values = np.arange(90, 96.2, 0.01)
    times = [ exponential(v - base_value) for v in auroc_values ]
    ax.plot(times, auroc_values, color='r')
    
    x,y = (4000, 95.02)
    name_bar = (x, x*7)
    ax.annotate('Eficiência = 1', (x*1.05,y+0.1), fontsize='small')
    #ax.annotate('Efficiency = 1', (x*1.05,y+0.1), fontsize='small')
    ax.plot(name_bar, (y,y), linewidth=.5, color='black')
    ax.plot((x/1.4, x), (y+0.45,y), linewidth=.5, color='black')
    
    plt.savefig('benchmark_auroc_efficiency')

    
def plot_acc_efficiency():

    summary_file = "benchmark_summary.csv"
    if not os.path.isfile(summary_file):
        extract_benchmark_summary()
    df = pd.read_csv(summary_file, index_col=0)
    
    #file = 'results_windows/ST_MrSEQL_FULL/results.csv'    
    #mr_seql = pd.read_csv(file)
    #mr_seql = mr_seql.set_index('strategy_name')
    #mr_seql = mr_seql.iloc[:,[1,0,2]]
    #mr_seql.iloc[:,:2] = mr_seql.iloc[:,:2]/100
    #mr_seql.columns = df.columns
    #df = pd.concat([df, mr_seql])
    
    
    avg_train_runtimes = df['runtime']
    avg_acc = df['accuracy']*100
    classifiers = avg_train_runtimes.index.values
    
    if (avg_train_runtimes.index.values != avg_acc.index.values).any():
        raise RuntimeError("The results doesnt match")
    
    
    fig, ax = plt.subplots(figsize=[8,6], dpi=200)
    
    colors = np.arange(avg_acc.size)
    scatter = ax.scatter(avg_train_runtimes,
                         avg_acc,
                         label=classifiers,
                         s = 70,
                         c=colors, cmap='viridis')
    
    handles, labels = scatter.legend_elements(prop="colors")
    #legend1 = ax.legend(handles, avg_train_runtimes.index.values ,
    #                    loc="lower right", title="Classes",
    #                    fancybox=True,
    #                    title_fontsize=14,
    #                    borderpad=1.0)
    
    
    for i, name in enumerate(classifiers):
        
        x = avg_train_runtimes[i]*1.22
        y = avg_acc[i]
        
        bar_len = (x, x*2.5)
        if name in ['HIVE-COTE v1.0','ProximityForest','InceptionTime']:
            bar_len = (x, x*5.6)
        elif name in ['RISE', 'BOSS','cBOSS']:
            bar_len = (x, x*2)
        elif name in ['STC', 'TSF']:
            bar_len = (x, x*1.6)
        
        if name == 'InceptionTime':
            y -= .35
            ax.annotate(name, (x,y), fontsize='small')
            y -= 0.03
            ax.plot((x, x*4.5), (y,y), linewidth=.5, color='black')
            ax.plot((avg_train_runtimes[i], x), (avg_acc[i],y), linewidth=.5, color='black')
        elif name in ['Catch22','BOSS']:
            y -= .25
            ax.annotate(name, (x,y), fontsize='small')
            y -= 0.03
            ax.plot(bar_len, (y,y), linewidth=.5, color='black')
            ax.plot((avg_train_runtimes[i], x), (avg_acc[i],y), linewidth=.5, color='black')
        else:
            y += .15
            ax.annotate(name, (x,y), fontsize='small')
            y -= 0.03
            ax.plot(bar_len, (y,y), linewidth=.5, color='black')
            ax.plot((avg_train_runtimes[i], x), (avg_acc[i],y), linewidth=.5, color='black')
    
    ax.set_title('Eficiência - Acurácia')
    ax.set_xlabel('média do tempo de treinamento (s)')
    ax.set_ylabel('média da acurácia (%)')
    #ax.set_title('Accuracy Efficiency')
    #ax.set_xlabel('Average fitting time (s)')
    #ax.set_ylabel('Average accuracy (%)')
    ax.set_xscale('log')
    ax.set_xlim(1100, 11**8)
    ax.grid(True, alpha=.2)
    ax.set_axisbelow(True)
    
    plt.savefig('benchmark_acc')
    
    

    def exponential(x):
        return 2**(x/4)
    
    ax.set_xlim(20, 12**8)
    base_value = 40.5545
    auroc_values = np.arange(74.5, 85.5, 0.01)
    times = [ exponential(v - base_value) for v in auroc_values ]
    ax.plot(times, auroc_values, color='r')
    
    x,y = (1450, 80)
    name_bar = (x, x*7)
    ax.annotate('Eficiência = 1', (x*1.05,y+0.1), fontsize='small')
    #ax.annotate('Efficiency = 1', (x*1.05,y+0.1), fontsize='small')
    ax.plot(name_bar, (y,y), linewidth=.5, color='black')
    ax.plot((x/1.4, x), (y+0.45,y), linewidth=.5, color='black')
    
    plt.savefig('benchmark_acc_efficiency')


if __name__ == '__main__':
    plot_auroc_efficiency()
    plot_acc_efficiency()











