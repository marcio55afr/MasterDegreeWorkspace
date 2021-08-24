

from sklearn.feature_selection import chi2





class WordRanking():
    """
        This class has differents functions to rank word based on its class
        separability. Where each feature is evaluated with a real number and 
        is mapped to this number as an array. The greater the number better
        is the feature.
    """    
    
    def chi2(sparse_matrix, labels):
        """
        This function only uses the sklearn.feature_selection.chi2 function
        and returns its return.

        """        
        return chi2(sparse_matrix, labels)