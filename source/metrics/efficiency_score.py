

def calculate_efficiency(results: pd.DataFrame, lowest_score: float = 0.5):
    """
        The results is a DataFrame containing score, fit runtime and predict
        runtime as columns in that order and each row represents the results
        of a different strategy or classifier.

        The lowest_score represents the score of a naive approach.
    """

    scores = results.iloc[:, 0]
    fit_runtime = results.iloc[:, 1]
    # predict_runtime = results.iloc[:,2]

    scores = scores - lowest_score
    efficiency = scores.apply(exponential) / fit_runtime
    # predict_rate = scores.apply(almost_exponential)/predict_runtime
    # efficiency = 4*fit_rate + predict_rate

    return efficiency  # np.around(efficiency).astype(np.int32)