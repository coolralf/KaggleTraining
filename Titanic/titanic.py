import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import os

path = os.getcwd()
data_train = pd.read_csv('./train.csv')
data_train
