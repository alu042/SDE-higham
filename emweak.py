#EMWEAK Test weak convergence of Euler-Maruyama
#
# SDE is dX = gamma*X dt + mu*X dW,  X(0) = Xzero
#      where gamma = 2, mu = 0.1, and Xzero = 1
#
# E-M uses 5 different timesteps: 2^(p-10),  p = 1,2,3,4,5.
# Examine weak convergence at T=1:  | E (X_L) - E (X(T)) |.
#
# Different paths are used for each E-M timestep.
# Adapted from 
# Desmond J. Higham "An Algorithmic Introduction to Numerical Simulation of 
#                    Stochastic Differential Equations"
#
# http://www.caam.rice.edu/~cox/stoch/dhigham.pdf


import numpy as np
import matplotlib.pyplot as plt
np.random.seed(102)

gamma=2; mu=0.1; Xzero=1; T=1
M=50000

Xem=np.zeros((5,1))
for p in range(1,6):
    Dt = 2**(p-10); L=float(T)/Dt
    Xtemp=Xzero*np.ones((M,1))
    for j in xrange(1,int(L)+1):
        Winc=np.sqrt(Dt)*np.random.randn(M)
        Xtemp += Dt*gamma*Xtemp + mu*np.multiply(Xtemp.T,Winc).T
    Xem[p-1] = np.mean(Xtemp,0)
Xerr = np.abs(Xem - np.exp(gamma))

Dtvals=np.power(float(2),[x-10 for x in range(1,6)])
plt.loglog(Dtvals,Xerr, 'b*-')
plt.loglog(Dtvals,Dtvals, 'r--')
plt.axis([1e-3, 1e-1, 1e-4, 1])
plt.xlabel('$\Delta t$'); plt.ylabel('| $E(X(T))$ - Sample average of $X_L$ |')
plt.title('emweak.py', fontsize=16)


### Least squares fit of error = C * Dt^q ###
A = np.column_stack((np.ones((p,1)), np.log(Dtvals))); rhs=np.log(Xerr)
sol = np.linalg.lstsq(A,rhs)[0]; q=sol[1][0]
resid=np.linalg.norm(np.dot(A,sol) - rhs)
#print 'q = ', q
#print 'residual = ', resid

plt.show()