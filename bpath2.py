#BPATH2  Brownian path simulation
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

dW=np.sqrt(dt)*np.random.randn(N)
W=np.cumsum(dW);
W=np.insert(W,0,0)

plt.plot(t,W)
plt.xlabel(r'$t$',fontsize=16); plt.ylabel(r'$W(t)$',fontsize=16,rotation=0)
plt.title('A Brownian path')
plt.show()

