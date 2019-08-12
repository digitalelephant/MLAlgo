import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
class NaiveBayes(object): 
    
    
    def fit(self, X, T, smoothing = 10e-3):
        self.gaussians = {}
        self.priors = {}
        
        labels = set(T)
        
        for c in labels:
            current_X = X[ T == c ]
            self.gaussians[c] = {
                'mean':current_X.mean(axis=0),
                'var':current_X.var(axis=0) + smoothing
            }
            self.priors[c] = float( len( T[ T==c ] ) )/len(T)
            
            
    def predict(self, X, T):
        
        N,D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N,K))
        
        for c,g in self.gaussians.iteritems():
            mean, var = g['mean'], g['var']
            P[:,c] = mvn.logpdf(X, mean=mean, cov=var) + np.log(self.priors[c])
            
        Y = np.argmax(P,axis=1)
        
        print 'Classification accuracy',(T == Y).mean()*100,'%'
