import pandas as pd

from source.experiments.hyperparam_tuning.hyperparam_tuning import run_3m_tuning_process

experiment_results = None  # used to get experiment results
exp = '2'
while exp:
    print(
        """Which experiment do you want run:
        
        1- Univariate Time Series Classification Benchmark
        2- 3M's hyperparameters tuning
        3- 
        0- Exit
"""
    )
    if exp == '0':
        break

    elif exp == '1':
        print('Not yet implemented')
        # run_guided_path()

    elif exp == '2':
        experiment_results = run_3m_tuning_process()
    else:
        print("Choose a valid option\n\n")

    break
