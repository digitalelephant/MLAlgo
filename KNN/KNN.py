import numpy as np
import matplotlib.pyplot as plt
from sortedcontainers import SortedList
from sklearn.utils import shuffle 

class KNN:
    def __init__(self,k):
        self.k = k
        
    def fit(self,X,T):
        self.X = X
        self.T = T
        
    #Lazy Classifer
    def predict(self,X,T):
        Y = np.zeros(len(X))
        for i, x in enumerate(X):
            sl = SortedList()
            for j, xt in enumerate(self.X):
                diff = x - xt
                dist = diff.dot(diff)
                if len(sl) < self.k:
                    sl.add( (dist,self.T[j]) )
                else:
                    if dist < sl[-1][0]:
                        del sl[-1]
                        sl.add( (dist,self.T[j]) )
             

            votes = {}
            for _,v in sl:
                votes[v] = votes.get(v,0) + 1
            
            max_votes = 0
            max_votes_class = -1
            for v, count in votes.iteritems():
                if count > max_votes:
                    max_votes = count
                    max_votes_class = v
            Y[i] = max_votes_class
        
        print 'Classification accuracy is',( Y == T ).mean()*100,'%' 
