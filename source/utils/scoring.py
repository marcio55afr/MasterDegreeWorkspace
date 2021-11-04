# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


    
def exponential(x):
    
    return 2**(x/4)
    #return 0

def scale(efficiency):    
    scaled_efficiency = [np.log(e) if e>=1 else 0 for e in efficiency]
    return scaled_efficiency
    
def calculate_efficiency(results: pd.DataFrame, lowest_score: float = 0.5):
    '''
        The results is a DataFrame containing score, fit runtime and predict
        runtime as columns in that order and each row represents the results
        of a different strategy or classifier.
        
        The lowest_score represents the score of a naive approach.
    '''
    
    scores = results.iloc[:,0]
    fit_runtime = results.iloc[:,1]
    #predict_runtime = results.iloc[:,2]
    
    scores = scores - lowest_score
    efficiency = scores.apply(exponential)/fit_runtime
    #predict_rate = scores.apply(almost_exponential)/predict_runtime
    #efficiency = 4*fit_rate + predict_rate
    
    return efficiency#np.around(efficiency).astype(np.int32)
    
def get_effiency_table():
        
    scores = [i*5+50 for i in range(10)]
    lowest_score = 50
    times = [2**(x/5) for x in np.arange(0,50,5)]            
    efficiency = [ exponential(scores[i]-lowest_score)/times[i] for i in range(10)]
    scaled_efficiency = scale(efficiency)
    
    table = pd.DataFrame(index=['Benchmark efficiency']*10)
    table['scores'] = scores
    table['fit runtime'] = times
    table['efficiency'] = efficiency
    
    return table