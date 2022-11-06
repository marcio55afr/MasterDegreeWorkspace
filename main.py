from source.experiments.variant_searching.variant_searching import run_guided_path

exp = '1'  # True
while exp:
    print(
        """Which experiment do you want run:
        
        1- Univariate Time Series Classification Benchmark
        2-
        3- 
        0- Exit
"""
    )
    if exp == '0':
        break

    elif exp == '1':
        run_guided_path()

    else:
        print("Choose a valid option\n\n")

    break
