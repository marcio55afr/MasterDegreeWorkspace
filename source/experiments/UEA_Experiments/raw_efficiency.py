

import matplotlib.pyplot as plt
import numpy as np




def plot_raw_efficiency():

    def exponential(x):
        return 2**(x/4)
    
    fig, ax = plt.subplots(figsize=[8,6], dpi=200)
    
    #ax.set_xlim(20, 12**8)
    base_value = 49.5794
    auroc_values = np.arange(70, 100, 0.01)
    times = [ exponential(v - base_value) for v in auroc_values ]
    ax.plot(times, auroc_values, color='r')
    
    #x,y = (4000, 95.02)
    #name_bar = (x, x*7)
    #ax.annotate('Eficiência = 1', (x*1.05,y+0.1), fontsize='small')
    #ax.plot(name_bar, (y,y), linewidth=.5, color='black')
    #ax.plot((x/1.4, x), (y+0.45,y), linewidth=.5, color='black')
    
    plt.savefig('efficiency_raw_plot')


if __name__ == '__main__':
    plot_raw_efficiency()
    