import os
import time
import wget
import zipfile

from source.data.config import *

univariate_ts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "Univariate2018_ts")

if not os.path.exists(univariate_ts_path):
    print("Downloading timeseries datasets...\nit can takes a few minutes")
    univariate_ts_zip = wget.download(UNIVARIATE_TS_LINK,
                                      out=os.path.dirname(os.path.abspath(__file__)))
    time.sleep(1)

    if not os.path.exists(univariate_ts_zip):
        raise f'Download failed!\nos.path.isfile({univariate_ts_zip} is equal to {os.path.isfile(univariate_ts_zip)})'
    else:
        print('Download completed!')

    print("Unzipping the archive...\n")
    with zipfile.ZipFile(univariate_ts_zip, 'r') as zip_ref:
        zip_ref.extractall(univariate_ts_path)
    os.remove(univariate_ts_zip)

    print("It's done!\n")
