# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def main():
    summary_file = "old/benchmark_summary.csv"
    if not os.path.isfile(summary_file):
        raise RuntimeError('The benchmark_summary.csv file must exist to procced')
    
    results = pd.DataFrame()
    for i in range(5):
        variant = f"V{i}_FULL/"
        test = 'ST_'+variant
        file = 'results_windows/'+test+'results.csv'
        
        df = pd.read_csv(file)
        df = df.set_index('strategy_name')
    
        results = pd.concat([results, df])
        
    results = results.iloc[:,[1,0,2]]
    results.index = ['V0',
                     'V1-KWS',
                     'V1-SG',
                     'V2',
                     'V3-NS',
                     'V3-NaR',
                     'V4']    
    
    runtime = results['fit runtime mean']
    
    auroc = results['AUROC mean']
    plot_efficiency_figure(auroc, runtime, 'auroc')
    acc = results['Accuracy mean']
    plot_efficiency_figure(acc, runtime, 'accuracy')
    
    
    
    mr_seql = 'results_windows/ST_MrSEQL_FULL/results.csv'    
    df = pd.read_csv(mr_seql)
    df = df.set_index('strategy_name')
    df = df.iloc[:,[1,0,2]]
    results = pd.concat([results, df])
    results.index = ['V0',
                     'V1-KWS',
                     'V1-SG',
                     'V2',
                     'V3-NS',
                     'V3-NaR',
                     'V4',
                     'MrSEQL']
    
        
        
    variant = f"V{3}_FULL/"
    test = 'ST_'+variant
    file = 'results_windows/'+test+'results.csv'
    results = pd.read_csv(file)
    results = results.set_index('strategy_name')
    results = results.iloc[:,[1,0,2]]
    results = pd.concat([results, df])
    results.index = ['3M-US',
                     '3M-MaR',
                     'MrSEQL']
    
    
    df = pd.read_csv(summary_file, index_col=0)
    df.columns = results.columns
    df.iloc[:,:2] = df.iloc[:,:2]*100
    
    results = pd.concat([results, df])
    
    
    runtime = results['fit runtime mean']
    auroc = results['AUROC mean']
    acc = results['Accuracy mean']
    
    plot_auroc_comparinson(auroc, runtime)
    plot_acc_comparinson(acc, runtime)




def plot_efficiency_figure(score, runtime, name='auroc', name_position=None):

    if (score.index.values != runtime.index.values).any():
        raise RuntimeError("The results doesnt match")
    if name_position is None:
        name_position = [-1]*score.size

    fig, ax = plt.subplots(figsize=[8,6], dpi=200)

    classifiers = score.index
    colors = np.arange(score.size)
    scatter = ax.scatter(runtime,
                         score,
                         label=classifiers,
                         s = 70,
                         c=colors, cmap='viridis')
    
    handles, labels = scatter.legend_elements(prop="colors")
    
    for i, strategy in enumerate(classifiers):
        
        x = runtime[i]*1.10
        y = score[i]
        name_bar = (x, x*1.3) if '-' in strategy else (x, x*1.15)
        
        
        if name_position[i] == 1:
            y += .2
            ax.annotate(strategy, (x/6,y), fontsize='small')
            y -= 0.2
            ax.plot(name_bar, (y,y), linewidth=.5, color='black')
            ax.plot((runtime[i], x/2), (score[i],y), linewidth=.5, color='black')
        else:
            y -= .3
            ax.annotate(strategy, (x,y), fontsize='small')
            y -= 0.2
            ax.plot(name_bar, (y,y), linewidth=.5, color='black')
            ax.plot((runtime[i], x), (score[i],y), linewidth=.5, color='black')
        
        if name == 'auroc':
            ax.set_title('Eficiência em AUROC por dataset')
            ax.set_xlabel('média do tempo de treinamento (s)')
            ax.set_ylabel('média da área sobre a curva ROC')
        else:
            ax.set_title('Eficiência em Acurácia por dataset')
            ax.set_xlabel('média do tempo de treinamento (s)')
            ax.set_ylabel('média da acurácia')
            
        ax.set_xscale('log')
        #ax.set_xlim(1100, 11**8)
        #ax.set_ylim(0.91, 0.965)
        #ax.add_artist(legend1)
        ax.grid(True, alpha=.2)
        ax.set_axisbelow(True)
    
    plt.savefig(f'{name}_efficiency_full')




def plot_auroc_comparinson(score, runtime):
    
    if (score.index.values != runtime.index.values).any():
        raise RuntimeError("The results doesnt match")

    fig, ax = plt.subplots(figsize=[8,6], dpi=200)

    classifiers = score.index
    colors = np.arange(score.size)
    scatter = ax.scatter(runtime,
                         score,
                         label=classifiers,
                         s = 70,
                         c=colors, cmap='viridis')
    
    handles, labels = scatter.legend_elements(prop="colors")
    
    for i, strategy in enumerate(classifiers):
        
        x = runtime[i]*1.30
        y = score[i]
        name_bar = (x, x*3)
        if strategy in ['HIVE-COTE v1.0', 'InceptionTime', 'ProximityForest']:
            name_bar = (x, x*12)
        elif strategy in ['V1-KWS','S-BOSS','TS-CHIEF']:
            name_bar = (x, x*4.5)
        elif strategy in ['V0','V2','V4','TSF']:
            name_bar = (x, x*2)
        
        
        if strategy in ['cBOSS']:
            name_bar = (x/6, x/2)
            y -= .25
            ax.annotate(strategy, (x/6,y), fontsize='small')
            y -= 0.15
            ax.plot(name_bar, (y,y), linewidth=.5, color='black')
            ax.plot((runtime[i], x/2), (score[i],y), linewidth=.5, color='black')
        
        elif strategy in ['WEASEL']:
            name_bar = (x/7, x/2)
            y += .45
            ax.annotate(strategy, (x/7,y), fontsize='small')
            y -= 0.15
            ax.plot(name_bar, (y,y), linewidth=.5, color='black')
            ax.plot((runtime[i], x/2), (score[i],y), linewidth=.5, color='black')
       
        elif strategy in ['InceptionTime']:
            name_bar = (x/17, x/2)
            y += .45
            ax.annotate(strategy, (x/19,y), fontsize='small')
            y -= 0.15
            ax.plot(name_bar, (y,y), linewidth=.5, color='black')
            ax.plot((runtime[i], x/2), (score[i],y), linewidth=.5, color='black')
            
        elif strategy in ['STC']:
            name_bar = (x/2.8, x/1.5)
            y += .45
            ax.annotate(strategy, (x/3,y), fontsize='small')
            y -= 0.15
            ax.plot(name_bar, (y,y), linewidth=.5, color='black')
            ax.plot((runtime[i], x/1.5), (score[i],y), linewidth=.5, color='black')
        
        elif strategy in ['BOSS']:
            y -= .8
            ax.annotate(strategy, (x,y), fontsize='small')
            y -= 0.2
            ax.plot(name_bar, (y,y), linewidth=.5, color='black')
            ax.plot((runtime[i], x), (score[i],y), linewidth=.5, color='black')
                    
        elif strategy in ['V1-SG','ROCKET','RISE','HIVE-COTE v1.0']:
            y += .0015
            ax.annotate(strategy, (x,y), fontsize='small')
            y -= 0.2
            ax.plot(name_bar, (y,y), linewidth=.5, color='black')
            ax.plot((runtime[i], x), (score[i],y), linewidth=.5, color='black')
        else:
            y -= .2
            ax.annotate(strategy, (x,y), fontsize='small')
            y -= 0.2
            ax.plot(name_bar, (y,y), linewidth=.5, color='black')
            ax.plot((runtime[i], x), (score[i],y), linewidth=.5, color='black')
        
    ax.set_title('Eficiência - AUROC')
    ax.set_xlabel('média do tempo de treinamento (s)')
    ax.set_ylabel('média da área sobre a curva ROC')
    #ax.set_title('AUROC Efficiency')
    #ax.set_xlabel('Average fitting time (s)')
    #ax.set_ylabel('Average area under the ROC curve')
            
    ax.set_xscale('log')
    ax.grid(True, alpha=.2)
    ax.set_axisbelow(True)

    plt.savefig('auroc_variants_vs_benchmarks')
    
    def exponential(x):
        return 2**(x/4)
    
    #ax.set_xlim(20, 12**8)
    base_value = 49.5794
    #auroc_values = np.arange(74, 96, 0.01)
    auroc_values = np.arange(83, 96, 0.01)
    times = [ exponential(v - base_value) for v in auroc_values ]
    ax.plot(times, auroc_values, color='r')
    
    x,y = (1000, 86.5)
    x,y = (1300, 88)
    name_bar = (x, x*10)
    ax.annotate('Eficiência = 1', (x*1.02,y+0.2), fontsize='small')
    #ax.annotate('Efficiency = 1', (x*1.02,y+0.2), fontsize='small')
    ax.plot(name_bar, (y,y), linewidth=.5, color='black')
    ax.plot((x/1.5, x), (y+0.5,y), linewidth=.5, color='black')
    
    plt.savefig('auroc_variants_vs_benchmarks_line')


def plot_acc_comparinson(score, runtime):
    
    if (score.index.values != runtime.index.values).any():
        raise RuntimeError("The results doesnt match")

    fig, ax = plt.subplots(figsize=[8,6], dpi=200)

    classifiers = score.index
    colors = np.arange(score.size)
    scatter = ax.scatter(runtime,
                         score,
                         label=classifiers,
                         s = 70,
                         c=colors, cmap='viridis')
    
    handles, labels = scatter.legend_elements(prop="colors")
    
    for i, strategy in enumerate(classifiers):
        
        x = runtime[i]*1.30
        y = score[i]
        name_bar = (x, x*3)
        if strategy in ['HIVE-COTE v1.0', 'InceptionTime', 'ProximityForest']:
            name_bar = (x, x*12)
        elif strategy in ['V1-KWS','S-BOSS','TS-CHIEF']:
            name_bar = (x, x*4.5)
        elif strategy in ['V0','V2','V4']:
            name_bar = (x, x*1.5)
        elif strategy in ['TSF','STC']:
            name_bar = (x, x*2)
        
        
        if strategy in ['cBOSS', 'BOSS']:
            name_bar = (x/6, x/2)
            y -= .3
            ax.annotate(strategy, (x/6,y), fontsize='small')
            y -= 0.15
            ax.plot(name_bar, (y,y), linewidth=.5, color='black')
            ax.plot((runtime[i], x/2), (score[i],y), linewidth=.5, color='black')
        
        elif strategy in ['WEASEL']:
            name_bar = (x/7, x/2)
            y += .45
            ax.annotate(strategy, (x/7,y), fontsize='small')
            y -= 0.15
            ax.plot(name_bar, (y,y), linewidth=.5, color='black')
            ax.plot((runtime[i], x/2), (score[i],y), linewidth=.5, color='black')
       
        elif strategy in ['InceptionTime']:
            name_bar = (x/17, x/2)
            y += .45
            ax.annotate(strategy, (x/20,y), fontsize='small')
            y -= 0.15
            ax.plot(name_bar, (y,y), linewidth=.5, color='black')
            ax.plot((runtime[i], x/2), (score[i],y), linewidth=.5, color='black')
                    
        elif strategy in ['V1-SG','S-BOSS']:
            y += .0015
            ax.annotate(strategy, (x,y), fontsize='small')
            y -= 0.2
            ax.plot(name_bar, (y,y), linewidth=.5, color='black')
            ax.plot((runtime[i], x), (score[i],y), linewidth=.5, color='black')
                    
        elif strategy in ['RISE','HIVE-COTE v1.0']:
            y += .6
            ax.annotate(strategy, (x,y), fontsize='small')
            y -= 0.2
            ax.plot(name_bar, (y,y), linewidth=.5, color='black')
            ax.plot((runtime[i], x), (score[i],y), linewidth=.5, color='black')
        else:
            y -= .3
            ax.annotate(strategy, (x,y), fontsize='small')
            y -= 0.2
            ax.plot(name_bar, (y,y), linewidth=.5, color='black')
            ax.plot((runtime[i], x), (score[i],y), linewidth=.5, color='black')
        
    ax.set_title('Eficiência em Acurácia por dataset')
    ax.set_xlabel('média do tempo de treinamento (s)')
    ax.set_ylabel('média da acurácia')
    #ax.set_title('Accuracy Efficiency')
    #ax.set_xlabel('Average fitting time (s)')
    #ax.set_ylabel('Average accuracy')
        
    ax.set_xscale('log')
    ax.grid(True, alpha=.2)
    ax.set_axisbelow(True)
    
    plt.savefig('accuracy_variants_vs_benchmarks')
    def exponential(x):
        return 2**(x/4)
    
    ax.set_xlim(20, 12**8)
    base_value = 40.5545
    auroc_values = np.arange(62, 86, 0.1)
    #auroc_values = np.arange(74, 86, 0.1)
    times = [ exponential(v - base_value) for v in auroc_values ]
    ax.plot(times, auroc_values, color='r')
    
    x,y = (400, 72.2)
    x,y = (1070, 78)
    name_bar = (x, x*10)
    ax.annotate('Eficiência = 1', (x*1.02,y+0.2), fontsize='small')
    #ax.annotate('Efficiency = 1', (x*1.02,y+0.2), fontsize='small')
    ax.plot(name_bar, (y,y), linewidth=.5, color='black')
    ax.plot((x/1.5, x), (y+0.5,y), linewidth=.5, color='black')
    
    plt.savefig('accuracy_variants_vs_benchmarks_line')




if __name__ == '__main__':
    main()













