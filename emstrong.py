# EMSTRONG Test strong convergence of Euler-Maruyama
#
# SDE is dX = gamma*X dt + mu*X dW,  X(0) = Xzero
#      where gamma = 2, mu = 1, and Xzero = 1
#
# Discretized Brownian path over [0,1] has dt = 2^(-9).
# E-M uses 5 different timesteps: 16dt, 8dt, 4dt, 2dt, dt.
# Examine strong convergence at T=1:  E | X_L - X(T) |.
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
T=1; N=2**9; dt = float(T)/N
M=1000

Xerr=np.zeros((M,5))
for s in xrange(M):
    dW=np.sqrt(dt)*np.random.randn(1,N)
    W=np.cumsum(dW)
    Xtrue = Xzero*np.exp((gamma-0.5*mu**2)*T+mu*W[-1])
    for p in range(5):
        R=2**p; Dt=R*dt; L=N/R
        Xem=Xzero
        for j in xrange(1,int(L)+1):
            Winc=np.sum(dW[0][range(R*(j-1),R*j)])
            Xem += Dt*gamma*Xem + mu*Xem*Winc
        Xerr[s,p]=np.abs(Xem-Xtrue)

Dtvals=dt*(np.power(2,range(5)))
plt.loglog(Dtvals,np.mean(Xerr,0),'b*-')
plt.loglog(Dtvals,np.power(Dtvals,0.5),'r--')
plt.axis([1e-3, 1e-1, 1e-4, 1])
plt.xlabel('$\Delta t$'); plt.ylabel('Sample average of $|X(T)-X_L|$')
plt.title('emstrong.py',fontsize=16)


### Least squares fit of error = C * Dt^q ###
A = np.column_stack((np.ones((5,1)), np.log(Dtvals))); rhs=np.log(np.mean(Xerr,0))
sol = np.linalg.lstsq(A,rhs)[0]; q=sol[1]
resid=np.linalg.norm(np.dot(A,sol) - rhs)
print 'residual = ', resid

plt.show()