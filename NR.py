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

NewtonR(1.2,1.2)


    