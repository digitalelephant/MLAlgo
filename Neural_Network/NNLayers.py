import numpy as np 
import matplotlib.pyplot as plt 

def sigmoid(a):
    return ( 1 / (1 + np.exp(-a) ) )

def softmax(a):
    expA = np.exp(a)
    return expA / np.sum(expA,axis=1,keepdims=True)

class NNLayers:

    

    def __init__(self,learning_rate,epoch,layer_sizes):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.layer_sizes = layer_sizes

    
    def fit(self,X,T):
        N,D = X.shape
        K = T.shape[1]

        # Feed forward


        S = {}
        for i in xrange(len(self.layer_sizes)+2):
            if i == 0:
                S[i] = D
            elif i == len(self.layer_sizes)+1:
                S[i] = K
            else:
                S[i] = self.layer_sizes[i-1]

        
        W = {}
        for i in xrange(len(self.layer_sizes)+1):
            W[i] = np.random.randn(S[i],S[i+1])
        
        Z = {}
        for i in xrange(len(self.layer_sizes)+2):
            if i == 0:
                Z[i] = X
            else:
                Z[i] = sigmoid(Z[i-1].dot(W[i-1]))

        Y = Z[len(self.layer_sizes)+1]
        print Y[0:2,:]

        error = {}
        derivative = {}
        
        # Back propagation
        
        i = len(self.layer_sizes)+1
        



    

