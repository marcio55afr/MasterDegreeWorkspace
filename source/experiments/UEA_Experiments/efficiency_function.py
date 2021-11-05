# -*- coding: utf-8 -*-

import os
import sys
sys.path.append('C:/Users/marci/Desktop/MasterDegreeWorkspace')
from source.utils import calculate_efficiency
from datasets.config import LARGER_DATASETS_NAMES

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
        
    scores = pd.Series(dtype=np.float32)
    for clf in benchmark_strategies:
        file = auroc_path + clf
        df = pd.read_csv(file, index_col=0)
        score = df.loc[LARGER_DATASETS_NAMES].mean(axis=1).mean()
        
        clf_name = clf.split('_')[0]
        scores.loc[clf_name] = score
    
    df = pd.DataFrame(index=scores.index)
    df.index.name = 'strategy'
    df['auroc'] = scores
    df['runtime'] = times
    df.to_csv('selected_benchmark_summary.csv')




def plot_eficiency():

    summary_file = "selected_benchmark_summary.csv"
    if not os.path.isfile(summary_file):
        extract_benchmark_summary()
    df = pd.read_csv(summary_file, index_col=0)
    
    avg_train_runtimes = df['runtime']
    avg_auroc = df['auroc']
    classifiers = avg_train_runtimes.index.values
    
    if (avg_train_runtimes.index.values != avg_auroc.index.values).any():
        raise RuntimeError("The results doesnt match")
    
    
    fig, ax = plt.subplots(figsize=[8,6], dpi=200)
    
    colors = np.arange(avg_auroc.size)
    scatter = ax.scatter(avg_train_runtimes,
                         avg_auroc,
                         label=classifiers,
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
        if name in ['TSF', 'ProximityForest', 'InceptionTime']:
            y -= .0025
            ax.annotate(name, (x,y), fontsize='small')
            y -= 0.0001
            ax.plot((x, x*1.85), (y,y), linewidth=1, color='black')
            ax.plot((avg_train_runtimes[i], x), (avg_auroc[i],y), linewidth=1, color='black')
        else:
            y += .0015
            ax.annotate(name, (x,y), fontsize='small')
            y -= 0.0002
            ax.plot((x, x*1.85), (y,y), linewidth=1, color='black')
            ax.plot((avg_train_runtimes[i], x), (avg_auroc[i],y), linewidth=1, color='black')
    
    ax.set_title('AUROC efficiency')
    ax.set_xlabel('fit runtime mean')
    ax.set_ylabel('roc auc mean')
    ax.set_xscale('log')
    ax.set_xlim(20, 10**8)
    #ax.set_ylim(0.85, 0.98)
    #ax.add_artist(legend1)
    ax.grid(True)
    
    
    def exponential(x):
        return 2**(x/4)
    
    base_value = 49.5794
    auroc_values = np.arange(86, 97, 0.01)
    times = [ exponential(v - base_value) for v in auroc_values ]
    print(times)
    ax.plot(times, auroc_values/100, color='r')
    
    plt.savefig('benchmark_auroc_eficiency')




if __name__ == '__main__':
    plot_eficiency()











