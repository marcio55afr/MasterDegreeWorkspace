

import pandas as pd


class ResolutionSelector():
    
    def get_best_resolution_max(ranked_words):
        
        maximum = ranked_words.groupby('resolution').max()
        n_reso = maximum.shape[0]
        best_reso = maximum.sort_values('rank', ascending=False).iloc[0:n_reso//2]
        return best_reso.index.values
        
        
    
    def get_best_resolution_mean(ranked_words):
        
        mean = ranked_words.groupby('resolution').mean()
        n_reso = mean.shape[0]
        best_reso = mean.sort_values('rank', ascending=False).iloc[0:n_reso//2]
        return best_reso.index.values
        
    
    def get_best_resolution_percentile(ranked_words):
                
        quantile = ranked_words.groupby('resolution').quantile(q=.95)
        n_reso = quantile.shape[0]
        best_reso = quantile.sort_values('rank', ascending=False).iloc[0:n_reso//2]
        return best_reso.index.values
        
    
    def get_best_resolution_top_mean(ranked_words):

        top_mean = pd.Series()
        for resolution in ranked_words['resolution'].unique():
            words = ranked_words.loc[ranked_words['resolution']==resolution, 'rank'].sort_values(ascending=False)
            n = words.shape[0]
            top_mean.loc[resolution] = words.iloc[0:n//10].mean()
            
        n_reso = top_mean.shape[0]
        best_reso = top_mean.sort_values(ascending=False).iloc[0:n_reso//2]
        return best_reso.index.values