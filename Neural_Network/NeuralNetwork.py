import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 



def sigmoid(a):
    return ( 1 / ( 1 + np.exp(-a) ) )

def softmax(a):
    return np.exp(a)/np.sum(np.exp(a))


class NeuralNetwork:
    def __init__(self,epoch,lr,hidden_layer_size):
        self.epoch = epoch
        self.lr  = lr
        self.hidden_layer_size = hidden_layer_size

    def fit(self,X,T):
        N,D = X.shape
        K = T.shape[1] # Ouput classes
       
        W = np.random.randn(D,self.hidden_layer_size)
        b = np.zeros((1,M))
       
        V = np.random.rand(self.hidden_layer_size,K)
        c = np.zeros((1,K))

        for i in xrange(self.epoch):
            
            Z = sigmoid(X.dot(W) + b)
            Y = softmax(Z.dot(V) + c)



    def predict(self,X,T):
        pass

