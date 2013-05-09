#STINT Approximate stochastic integrals
#
# Ito and Stratonovich integrals of W dW
#
# Adapted from 
# Desmond J. Higham "An Algorithmic Introduction to Numerical Simulation of 
#                    Stochastic Differential Equations"
#
# http://www.caam.rice.edu/~cox/stoch/dhigham.pdf

import numpy as np
import math

np.random.seed(100)
T=1; N=500; dt=float(T)/N

dW=math.sqrt(dt)*np.random.randn(1,N)
W=np.cumsum(dW)
W = np.insert(W,0,0)

ito=np.sum(W[0:-1]*dW)
strat=np.sum((0.5*(W[0:-1]+W[1:])+0.5*math.sqrt(dt)*np.random.randn(1,N))*dW)

itoerr=np.abs(ito-0.5*(W[-1]**2-T))
straterr=np.abs(strat-0.5*W[-1]**2)

print "Ito: ", ito
print "Strat: ", strat
print "Error of Ito: ",itoerr
print "Error of Strat: ", straterr
