import numpy as np
import matplotlib.pyplot as plt
from sortedcontainers import SortedList
from sklearn.utils import shuffle 
from NaiveBayes import NaiveBayes
      

import pandas as pd
df = pd.read_csv('../NaiveBayes/train_digits.csv')
df = shuffle(df)
df.reset_index(drop=True,inplace=True)
# df = df.iloc[0:10000]
df.head()
Y = df['label']
df.drop(['label'],inplace=True,axis=1)
X = df.values
cutoff = int(len(df)*0.75)
Xtrain = X[:cutoff,:]
Ttrain = Y[:cutoff]
Xtest = X[-cutoff:,:]
Ttest = Y[-cutoff:]

nb = NaiveBayes()
nb.fit(Xtrain, Ttrain)
nb.predict(Xtest,Ttest)
