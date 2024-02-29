import os

import numpy as np
import pandas as pd


df = pd.read_csv('hbku2019/labels/labels/labels_train.csv',header=None)
labels = df.iloc[:, 0].tolist()
labels = df.iloc[:, 1:].to_numpy()
for i in labels:
    print(i)

