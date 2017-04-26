#%%
import pandas as pd
import numpy as np
import os
from pandas import Series,DataFrame

path = os.getcwd()
data_train = pd.read_csv(r"Titanic/train.csv")
print(data_train)