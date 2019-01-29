import numpy as np 
import matplotlib.pyplot as plt 

def sigmoid(a):
    return ( 1 / (1 + np.exp(-a) ) )

def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1,keepdims=True)

class NNLayers:

    

    def __init__(self,learning_rate,epoch,layer_sizes):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.layer_sizes = layer_sizes

    
    def fit(self,X,T):
        N,D = X.shape
        K = T.shape[1]



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
        
        cost = {}
        k = 0 

        for e in xrange(self.epoch):
            # Feed forward
        
            Z = {}
            for i in xrange(len(self.layer_sizes)+2):
                if i == 0:
                    Z[i] = X
                elif i == len(self.layer_sizes)+1:
                    Z[i] = softmax(Z[i-1].dot(W[i-1]))
                else:
                    Z[i] = sigmoid(Z[i-1].dot(W[i-1]))

            Y = Z[len(self.layer_sizes)+1]

            
            # Back propagation
            
            error = {}
            derivative = {}
            
            
            i = len(self.layer_sizes)+1

            while i > 0:
                if i == len(self.layer_sizes)+1:
                    error[i] = Z[i]-T
                    derivative[i-1] = Z[i-1].T.dot(error[i])
                    W[i-1] -= (self.learning_rate/N)*derivative[i-1]
                else:
                    error[i] = error[i+1].dot(W[i].T)*Z[i]*(1-Z[i])
                    derivative[i-1] = Z[i-1].T.dot(error[i])
                    W[i-1] -= (self.learning_rate/N)*derivative[i-1]
                i -= 1
            
            J = (-1)* np.sum(T*np.log(Y)) / N            
   
            if e%100 == 0:
                print 'Cost is ',J
                cost[k] = J
                k += 1
        
        self.W = W  

        plt.plot(cost.keys(),cost.values())
        plt.show()

    def predict(self,X,T):
        prev = X
        for i in xrange(len(self.W)):
            if i == len(self.W)-1:
                prev = softmax(prev.dot(self.W[i]))
            else:
                prev = sigmoid(prev.dot(self.W[i]))
        
        Y = np.zeros((T.shape[0],T.shape[1]))
        Y[np.arange(T.shape[0]),prev.argmax(axis=1)] = 1
        print "Classification rate is",(T == Y).mean()*100,"%"
        