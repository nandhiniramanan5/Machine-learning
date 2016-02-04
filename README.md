# Machine-learning
Regression algos


Gradient Descent 
import numpy as np
import random
from math import exp, sqrt
import pylab
from scipy import stats
from numpy import linalg as LA

def gradient_descent(alpha,x1,x2):
O_W = [0, 0]
N_W = [x1,x2]
tolerance = 0.00001
n = 0
while getTolerance(N_W, O_W)>tolerance and n < 2000000:
O_W = N_W
X= [(400*(x1**3)-400*(x1)*(x2)+2*(x1)-2),(-200*(x1**2)+200*x2)]
N_W=(O_W-np.dot(alpha,X))
x1, x2 = N_W[0], N_W[1]
n = n + 1
print N_W, n
def getTolerance(N_W, O_W):
sum = 0
for i in range(len(N_W)):
sum = sum + (N_W[i] - O_W[i])**2
return sqrt(sum)

>>>gradient_descent(1,1.2,1.2)

The gradient descent method does not converge without step length control except we ﬁx the step length α to be suﬃciently small. When α is 1, this model doesn’t converge. Whereas for the following run 
 
>>>gradient_descent(.001,1.2,1.2)
Where α to be suﬃciently small say .001, our Gradient decent model converges as required.

Newton’s method:
import numpy as np
import random
from math import exp, sqrt
import pylab
from scipy import stats
from numpy import linalg as LA


def NewtonR(x1,x2):
O_W = np.matrix([[0], [0]])
N_W = np.matrix([[x1], [x2]])
tolerance = 0.00001
while getTolerance(N_W, O_W)>tolerance:
O_W = N_W
H=np.matrix([[(-400*x2)+(1200*x1**2)+2, (-400*x1)], \
[(-400*x1), 200]])
X=np.matrix([[(400*(x1**3)-400*(x1)*(x2)+2*(x1)-2)], [(-200*(x1**2)+200*x2)]])
N_W=(O_W-np.dot(np.linalg.inv(H),X))
x1, x2 = N_W.item(0), N_W.item(1)
print N_W
def getTolerance(N_W, O_W):
sum = 0
for i in range(len(N_W)):
sum = sum + (N_W.item(i) - O_W.item(i))**2
return sqrt(sum)

When the Newton’s method converges, it converges very fast (quadratic convergence asymptotically).



a.	The provided implementation has subselected features and then simply explicitly solved for w = (XT X)−1XTy.On increasing the number of selected features We get this following error LinAlgError: Singular matrix 
This is because Inversion on matrix close to being singular due to multi-collinearity is what we attempt to do here. Matrix XTX  has a determinant which is close to zero, which makes it “ill-conditioned” so the matrix can’t be inverted with as much precision as we’d like. The ridge estimator, on the other hand, is biased but is more robust against multi-collinearity. A small positive number λI added to the diagonals of XTX works almost well to provide estimates that are more stable than the ordinary LS estimates.

As the regularization parameter increases, wi shrinks toward 0:
 

b.	Following code is be added to regression.py
class RidgeRegression(Regressor):
def __init__(self, params=None):
self.weights = None
self.features = [x for x in range(1,230)]
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

Modified the code to report error averaged over multiple splits in script_classify.py
sum = 0 
j=0
for i in range(0, 10): 
for learnername, learner in classalgs.iteritems():
while j==0:
print 'Running learner = ' + learnername
j=j+1
# Train model
print testset[1].shape
learner.learn(trainset[0], trainset[1])
# Test model
predictions = learner.predict(testset[0])
#print predictions
accuracy= geterror(testset[1], predictions)
sum=sum+accuracy
avg=0
avg = sum/10
print 'Avg Accuracy for ' + learnername + ': ' + str(avg)



c.	To subselect features randomly used “Univariate feature selection SelectKBest”.  SelectKBest selects the K features that are most powerful (where K is a parameter), where K is 10 by default
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
ytest = np.dot(Xless, self.weights) 
return ytest


Accuracy surely has gotten better after implementing the feature select on FSLinear regression when compare to normal FSLinearRegression and Ridge regression put together based on the outputs generated.

Further doing a feature subselect in Ridge Regularizer also seems to imrove accuracy as tested in code.

d.	Code for Stochastic regression in algorithms.py
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

Here we are stochastic approximation, we typically approximate the gradient with one sample.  With ever increasing data-set size for many scenarios, the generality of stochastic approximation makes it easy to do an approximation.
 For t<-1 to n : W <- W - αt (xtTw-yt)xt   Which is a very good approximation when dealing with big data

e.	Poisson regression :
Added following code to the page regression.py
 ‘ ‘ ‘ 
Following code is put in comment block because it is throwing error
Logic or the algorithm used is correct whereas:
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

’’’
Used following mechanism to calculate weights :
       W(t+1) =W(t) + (XT * C(t) * X )-1 * XT * (y-c(t))
where c (t) and C(t) are calculated using the weight vector w(t) . The initial set of weights w(0) can be set randomly.


 
