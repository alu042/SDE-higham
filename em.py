#EM Euler-Maruyama method on a linear SDE
#
# SDE is dX = gamma*X dt + mu*X dW, X(0) = Xzero
#      where gamma = 2, mu = 1, and Xzero = 1
#
# Discretized Brownian path over [0,1] has dt = 2^(-8).
# Euler-Maruyama uses timestep R*dt
#
# Adapted from 
# Desmond J. Higham "An Algorithmic Introduction to Numerical Simulation of 
#                    Stochastic Differential Equations"
#
# http://www.caam.rice.edu/~cox/stoch/dhigham.pdf

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(100)

gamma=2; mu=1; Xzero=1
T=1; N=2**8; dt = float(T)/N
t=np.linspace(0,T,N+1)

dW=np.sqrt(dt)*np.random.randn(1,N)
W=np.cumsum(dW)

Xtrue=Xzero*np.exp((gamma-0.5*mu**2)*t[1:]+mu*W); Xtrue=np.insert(Xtrue,0,Xzero)
ax=plt.subplot(111)
ax.plot(t,Xtrue)

R=4; Dt=R*dt; L=float(N)/R
Xem=np.zeros(L+1); Xem[0] = Xzero

for j in xrange(1,int(L)+1):
    Winc=np.sum(dW[0][range(R*(j-1),R*j)])
    Xem[j] = Xem[j-1] + Dt*gamma*Xem[j-1] + mu*Xem[j-1]*Winc

emerr=np.abs(Xem[-1]-Xtrue[-1])
print "error: ", emerr

ax.plot(np.linspace(0,T,T/Dt+1),Xem,'r--*')
ax.legend(("exact","em"),loc=2)
plt.show()
