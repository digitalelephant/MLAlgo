import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.utils import shuffle 
from LogisticRegression import LogisticRegression


df = pd.read_csv('../Logistic_Regression/train.csv')
print df.columns
df = df[['Pclass','Sex','Age','SibSp','Parch','Survived']]

df = shuffle(df)
df['Age'].fillna(df['Age'].mean(),inplace =True)
df['SibSp'].fillna(df['SibSp'].mean(),inplace =True)
df['Parch'].fillna(df['Parch'].mean(),inplace =True)
df['Pclass'].fillna(df['Pclass'].mean(),inplace =True)

gender_dict = {'male':1,'female':0}
df['Sex'] = df['Sex'].map(gender_dict)

cutoff = int(len(df)*0.8)
Xtrain = df.values[:cutoff,:-1]
Ttrain = df.values[:cutoff,-1:]
Xtest = df.values[-cutoff:,:-1]
Ttest = df.values[-cutoff:,-1:]

lr = LogisticRegression(4000,0.00001,0.00003,'L2')
lr.fit(Xtrain,Ttrain)
lr.predict(Xtest,Ttest)
