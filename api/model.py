import pandas as pd

from information_retrieval.utils.datasets import DrugsInformation
from information_retrieval import config

def get_DI(idx):
    information = DrugsInformation(pd.read_parquet(config['PATH']['PROCESSED_DATA_PATH']))[idx]
    return information