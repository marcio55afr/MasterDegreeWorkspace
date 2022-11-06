from setuptools import setup, find_packages

setup(
    name='4T/3M experiment',
    version='1.0',
    long_description=__doc__,
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'pandas==1.4.4',
        'matplotlib==3.5.3',
        'scikit-learn==1.1.2',
        'sktime==0.13.2',
        'wget==3.2'
    ]
)