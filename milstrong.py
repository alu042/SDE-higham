#MILSTRONG Test strong convergence of Milstein
#
# SDE is  dX = r*X*(K-X) dt + beta*X dW,   X(0) = Xzero,
#       where r = 2, K = 1, beta = 0.25, Xzero = 0.5.
#
# Discretized Brownian path over [0,1] has dt = 2^(-11).
# Milstein uses timesteps 128*dt, 64*dt, 32*dt, 16*dt (also dt for reference).
#
# Examines strong convergence at T=1:  E | X_L - X_T |.
#
# Adapted from 
# Desmond J. Higham "An Algorithmic Introduction to Numerical Simulation of 
#                    Stochastic Differential Equations"
#
# http://www.caam.rice.edu/~cox/stoch/dhigham.pdf

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)
r=2; K=1; beta=0.25; Xzero=0.5
T=1; N=2**11; dt=float(T)/N
M=500
R = [1, 16, 32, 64, 128]

dW = np.sqrt(dt)*np.random.randn(M,N)
Xmil = np.zeros((M,5))
for p in range(5):
    Dt = R[p]*dt; L=float(N)/R[p]
    Xtemp=Xzero*np.ones(M)
    for j in xrange(1,int(L)+1):
        Winc=np.sum(dW[:,range(R[p]*(j-1),R[p]*j)],axis=1)
        Xtemp += Dt*r*Xtemp*(K-Xtemp) + beta*Xtemp*Winc \
                 + 0.5*beta**2*Xtemp*(np.power(Winc,2)-Dt)
    Xmil[:,p] = Xtemp

Xref = Xmil[:,0]
Xerr = np.abs(Xmil[:,range(1,5)] - np.tile(Xref,[4,1]).T)
Dtvals = np.multiply(dt,R[1:5])

plt.loglog(Dtvals,np.mean(Xerr,0),'b*-')
plt.loglog(Dtvals,Dtvals,'r--')
plt.axis([1e-3, 1e-1, 1e-4, 1])
plt.xlabel('$\Delta t$'); plt.ylabel('Sample average of $|X(T)-X_L|$')
plt.title('milstrong.py',fontsize=16)

#### Least squares fit of error = C * Dt^q ####
A = np.column_stack((np.ones((4,1)), np.log(Dtvals)))
rhs=np.log(np.mean(Xerr,0))
sol = np.linalg.lstsq(A,rhs)[0]; q=sol[1]
resid=np.linalg.norm(np.dot(A,sol) - rhs)
print 'residual = ', resid

plt.show()
