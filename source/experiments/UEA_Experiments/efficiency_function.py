# -*- coding: utf-8 -*-

import os
import sys
sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace')
from source.utils import calculate_efficiency
from source.experiments.UEA_Experiments.datasets.config import LARGER_DATASETS_NAMES

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




def plot_eficiency():

    summary_file = "benchmark_summary.csv"
    if not os.path.isfile(summary_file):
        extract_benchmark_summary()
    df = pd.read_csv(summary_file, index_col=0)
    
    avg_train_runtimes = df['runtime']
    avg_auroc = df['auroc']
    #avg_acc = df['accuracy']
    #avg_auroc = avg_acc
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
        
        x = avg_train_runtimes[i]*1.15
        y = avg_auroc[i]
        if name == 'InceptionTime':
            y -= .0035
            ax.annotate(name, (x,y), fontsize='small')
            y -= 0.0002
            ax.plot((x, x*1.85), (y,y), linewidth=.5, color='black')
            ax.plot((avg_train_runtimes[i], x), (avg_auroc[i],y), linewidth=.5, color='black')
        elif name in ['Catch22','BOSS']:
            y -= .0025
            ax.annotate(name, (x,y), fontsize='small')
            y -= 0.0002
            ax.plot((x, x*1.85), (y,y), linewidth=.5, color='black')
            ax.plot((avg_train_runtimes[i], x), (avg_auroc[i],y), linewidth=.5, color='black')
        else:
            y += .0015
            ax.annotate(name, (x,y), fontsize='small')
            y -= 0.0002
            ax.plot((x, x*1.85), (y,y), linewidth=.5, color='black')
            ax.plot((avg_train_runtimes[i], x), (avg_auroc[i],y), linewidth=.5, color='black')
    
    ax.set_title('Eficiência - AUROC')
    ax.set_xlabel('média do tempo de treinamento (s)')
    ax.set_ylabel('média da área sobre a curva ROC')
    ax.set_xscale('log')
    ax.set_xlim(1100, 11**8)
    #ax.set_ylim(0.91, 0.965)
    #ax.add_artist(legend1)
    ax.grid(True, alpha=.2)
    ax.set_axisbelow(True)
    
    #plt.savefig('benchmark_auroc')
    
    
    def exponential(x):
        return 2**(x/4)
    
    ax.set_xlim(20, 12**8)
    base_value = 49.5794
    auroc_values = np.arange(89.5, 96.2, 0.01)
    times = [ exponential(v - base_value) for v in auroc_values ]
    ax.plot(times, auroc_values/100, color='r')
    
    plt.savefig('benchmark_auroc_eficiency')




if __name__ == '__main__':
    plot_eficiency()











