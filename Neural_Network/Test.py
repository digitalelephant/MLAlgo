import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle 
from NeuralNetwork import NeuralNetwork
from NNLayers import NNLayers


df = pd.read_csv('train.csv')

Y = np.zeros((len(df),10))

df = shuffle(df)

Y[np.arange(len(df)),df['label'].values] = 1

df.drop(['label'],inplace=True,axis=1)

X = df.values

cutoff = int(len(df)*0.8)
Xtrain = X[:cutoff,:]
Ttrain = Y[:cutoff,:]
Xtest = X[-cutoff:,:]
Ttest = Y[-cutoff:,:]

# df = pd.read_csv('../Logistic_Regression/train.csv')
# print df.columns
# df = df[['Pclass','Sex','Age','SibSp','Parch','Survived']]

# df = shuffle(df)
# df['Age'].fillna(df['Age'].mean(),inplace =True)
# df['SibSp'].fillna(df['SibSp'].mean(),inplace =True)
# df['Parch'].fillna(df['Parch'].mean(),inplace =True)
# df['Pclass'].fillna(df['Pclass'].mean(),inplace =True)
# df['Age'] = (df['Age']-df['Age'].mean())/df['Age'].std()
# df['SibSp'] = (df['SibSp']-df['SibSp'].mean())/df['SibSp'].std()
# df['Parch'] = (df['Parch']-df['Parch'].mean())/df['Parch'].std()
# df['Pclass'] = (df['Pclass']-df['Pclass'].mean())/df['Pclass'].std()

# gender_dict = {'male':1,'female':0}
# df['Sex'] = df['Sex'].map(gender_dict)

# cutoff = int(len(df)*0.8)
# Xtrain = df.values[:cutoff,:-1]
# Ttrain = df.values[:cutoff,-1:]
# Xtest = df.values[-cutoff:,:-1]
# Ttest = df.values[-cutoff:,-1:]

# lr = NeuralNetwork(2500,0.03,64)
# lr.fit(Xtrain,Ttrain)
# lr.predict(Xtest,Ttest)

nn = NNLayers(0.03,2500,np.array([10,10,5]))
nn.fit(Xtrain,Ttrain)