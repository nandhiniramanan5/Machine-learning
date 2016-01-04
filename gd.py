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
    
gradient_descent(.001,-1.2,1)


    