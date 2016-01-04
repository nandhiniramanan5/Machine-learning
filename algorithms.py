import numpy as np
import utilities as utils
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
collen = 0
class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects value between max and min in training set.
    """
    
    def __init__( self, params=None ):
        """ Params can contain any useful parameters for the algorithm """
        self.min = 0
        self.max = 1
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        return ytest

class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """
    def __init__( self, params=None ):
        self.mean = None
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.mean = np.mean(ytrain)
        
    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean

class FSLinearRegression(Regressor):
    """
    Univariate feature selection SelectKBest
    """
    def __init__( self, params=None ):
        self.weights = None
        self.features = [x for x in range (1,60)] 
       
        
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        #Xless = Xtrain[:,self.features]
        iris = load_iris()
        Xtrain, ytrain = iris.data, iris.target
        Xtrain.shape        
        Xless = SelectKBest(chi2, k='all').fit_transform(Xtrain, ytrain)
        collen = Xless.shape[1]
        self.features = [x for x in range(1,collen+1)]
        #print Xless.shape
       # IdentityMAt = np.identity(length)
        
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)), Xless.T),ytrain)
                
    def predict(self, Xtest):
        Xless = Xtest[:,self.features]        
        ytest = np.dot(Xless, self.weights.T)       
        return ytest
        
        
class StochasticGradient(Regressor):
    
    def __init__( self, params=None ):
        self.weights = None
        self.features = [x for x in range (0,230)] 
       
        
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        Xless = Xtrain[:,self.features] 
        NewWeights = np.ndarray((1,230))
        #print NewWeights.shape
        alpha = 1.0
        for i in range(1, 100):
            self.weights = NewWeights
            NewWeights = self.weights - alpha * (Xless[i].T * self.weights-ytrain[i]) * Xless[i]
        #self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)), Xless.T),ytrain)
        
    def predict(self, Xtest):
        Xless = Xtest[:,self.features]        
        ytest = np.dot(Xless, self.weights)       
        return ytest

class RidgeRegression(Regressor):
    def __init__(self, params=None):
        self.weights = None
        self.features = [x for x in range(1,10)]
        
    def learn(self, Xtrain,ytrain):
        _lambda = 150
        Xless = Xtrain[:,self.features]
        length = len(self.features)
        IdentityMAt = np.identity(length)
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)+np.dot(_lambda,IdentityMAt)), Xless.T),ytrain)
            
    def predict(self, Xtest):
        Xless = Xtest[:,self.features]        
        ytest = np.dot(Xless, self.weights)       
        return ytest
'''  
class PoissonRegression(Regressor):
    def __init__(self, params=None):
        self.weights = None
        self.features = [x for x in range(1,10)]
        
    def learn(self, Xtrain,ytrain):
        NewWeights = np.ndarray((1,230))
        Xless = Xtrain[:,self.features]
        alpha = 1.0
        for i in range(1, len(Xless))
        c(i) = exp(np.dot(self.weights.T,Xless(i))
        
        self.weights = NewWeights
        newweights = self.weights - alpha * np.linalg.inv(Xless.T * C.T * Xless) * Xless.T * (ytrain - c(i))
            
    def predict(self, Xtest):
        Xless = Xtest[:,self.features]        
        ytest = np.dot(Xless, self.weights)       
        return ytest
'''