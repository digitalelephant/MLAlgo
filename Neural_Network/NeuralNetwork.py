import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 



def sigmoid(a):
    return ( 1 / ( 1 + np.exp(-a) ) )

def softmax(a):
    return np.exp(a)/np.sum(np.exp(a),axis=1,keepdims=True)


class NeuralNetwork:
    def __init__(self,epoch,lr,hidden_layer_size):
        self.epoch = epoch
        self.lr  = lr
        self.hidden_layer_size = hidden_layer_size

    def fit(self,X,T):
        N,D = X.shape
        K = T.shape[1] # Output classes
        W = np.random.randn(D,self.hidden_layer_size)
        b = np.zeros((1,self.hidden_layer_size))
        V = np.random.randn(self.hidden_layer_size,K)
        c = np.zeros((1,K))
        cost = {}
        k = 0

        for i in xrange(self.epoch):
            
            Z = sigmoid(X.dot(W) + b)
            # Y = sigmoid(Z.dot(V) + c) # Only one output class
            Y = softmax(Z.dot(V) + c) # For multiclass classification 
            # J = (-1)*( T*np.log(Y) + (1-T)*np.log(1-Y ) ).sum() / N # Cost for binary classification
            J = (-1)* np.sum(T*np.log(Y)) / N # Cost for multiclass classification
            if i%100 == 0:
                print "Cost = ",J
                cost[k] = J
                k = k+1

            
            # delta_output = (Y-T)*Y*(1-Y) # For sigmoid output
            delta_output = Y-T # For softmax output
            delta_bias_c = delta_output.sum(axis=0)
            V = V - (self.lr/N)*(Z.T.dot(delta_output))
            c = c - (self.lr/N)*delta_bias_c


            delta_hidden = (Y-T).dot(V.T)*Z*(1-Z)
            delta_bias_b = delta_hidden.sum(axis=0)
            W = W - (self.lr/N)*(X.T.dot(delta_hidden))
            b = b - (self.lr/N)*delta_bias_b

        self.W = W
        self.b = b
        self.V = V
        self.c = c

        plt.plot(cost.keys(),cost.values())
        plt.show()

            

    def predict(self,X,T):
        Z = sigmoid(X.dot(self.W) + self.b)
        # Type one 
        # For binary classification
        # output = sigmoid(Z.dot(self.V) + self.c)
        # Y = np.round(output)
        
        # Type two
        # For multiclass classification
        output = softmax(Z.dot(self.V) + self.c)
        Y = np.zeros((T.shape[0],T.shape[1]))
        Y[np.arange(T.shape[0]),output.argmax(axis=1)] = 1
        print "Classification rate is",(T == Y).mean()*100,"%"
        

