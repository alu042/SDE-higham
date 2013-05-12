#BPATH3 Function along a Brownian path
#
# Adapted from 
# Desmond J. Higham "An Algorithmic Introduction to Numerical Simulation of 
#                    Stochastic Differential Equations"
#
# http://www.caam.rice.edu/~cox/stoch/dhigham.pdf

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)
T=1; N=500; dt=float(T)/N; 
t=np.linspace(0,T,N+1)

M=1000
dW=np.sqrt(dt)*np.random.randn(M,N)
W=np.cumsum(dW,1)
U=np.exp(np.tile(t[1:],(M,1))+0.5*W)
Umean=np.mean(U,axis=0)
Umean=np.insert(Umean,0,1)


plt.plot(t,Umean)
for i in range(5):
    plt.plot(t,np.concatenate(([1,],U[i,:])), 'r--')
plt.legend(('mean of 1000 paths', '5 individual paths'),loc=2,shadow=True)
plt.xlabel('$t$',fontsize=16); plt.ylabel('$U(t)$',fontsize=16,rotation=0)

averr = np.linalg.norm(Umean-np.exp(9*t/8),np.inf)
print 'averr = ', averr

plt.show()

