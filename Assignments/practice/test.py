import pandas as pd
import numpy as np
from mf import MF


df_train = pd.read_csv('all/train.csv')
df_train = df_train[0:10000]
R= np.array(df_train.pivot(index = 'User', columns ='Track', values = 'Rating').fillna(0))
d_mf = MF(R, K=20, alpha=0.001, beta=0.01, iterations=100)
training_process = d_mf.train()
print()
print("P x Q:")
print(d_mf.full_matrix())
print()