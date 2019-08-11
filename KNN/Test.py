import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.utils import shuffle 
from KNN import KNN

df = pd.read_csv('../KNN/train_digits.csv')

df.head()

df = shuffle(df)

df.reset_index(drop=True,inplace=True)

df.head()

Y = df['label']
df.drop(['label'],inplace=True,axis=1)
X = df.values



cutoff = int(len(df)*0.8)
Xtrain = X[:cutoff,:]
Ttrain = Y[:cutoff]
Xtest = X[-cutoff:,:]
Ttest = Y[-cutoff:]


knn = KNN(1000)
knn.fit(Xtrain,Ttrain)
knn.predict(Xtest,Ttest)
