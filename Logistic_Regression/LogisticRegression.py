import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


def sigmoid(a):
    return (1/(1+np.exp(-a)))




class LogisticRegression:
    def __init__(self,epoch,lr,lam,reg_type):
        self.epoch = epoch
        self.lr = lr
        self.lam = lam
        self.reg_type = reg_type



    def forward(self,X,W):
        return sigmoid(X.dot(W))

    def fit(self,X,T):
        print "Shape of T",T.shape

        N,D = X.shape
        W = np.random.rand(D,1)
        cross_entropy = {}
        k = 0
        
        for i in xrange(self.epoch):
            
            Y = self.forward(X, W)
            
            J = (-1)*( T*np.log(Y) + (1-T)*np.log(1-Y ) ).sum() / N
            W = W - (self.lr)*(X.T.dot(Y-T))

            if self.reg_type == 'L2':
                J = J + (self.lam/2)*(W*W).sum()
                W = W + self.lam*(W)
            elif self.reg_type == 'L1':
                J = J + self.lam*(W).sum()
                W = W + self.lam*(W/abs(W))
            else:
                J = J + 0
                W = W + 0    

            
            if  i%100 == 0:
                cross_entropy[k] = J
                print "Cross Entropy", cross_entropy[k]
                k = k+1
        
        self.W = W
        
        plt.plot(cross_entropy.keys(),cross_entropy.values())
        plt.show()

    def predict(self,X,T):
        Y = np.round( self.forward(X,self.W) )
        print "Classification Accuracy = ",( np.mean( Y == T) * 100)


        