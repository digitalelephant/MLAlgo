import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 


def softmax(a):
    return np.exp(a)/np.sum(np.exp(a))

b = softmax(np.array([1,1,1]))
print softmax(b)

a = np.array([[1,1],[2,2],[3,3]])
print a
b = np.array([4,4])
print (a+b)