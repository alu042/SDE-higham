# Milstein method applied to a 3-dimensional SDE
#
# SDE:  dX(1) = X(1) X(2) dW(1),                     X(1)_0 = 1
#       dX(2) = -(X(2) - X(3)) dt + 0.3 X(2) dW(2),  X(2)_0 = 0.1
#       dX(3) = (X(2) - X(3)) dt,                    X(3)_0 = 0.1
#
# Discretized Brownian path over [0,1] has delta = 2^(-18)
# Milstein timestep is Delta = sqrt(delta)
# Substeps for double integral are of size delta
#
# Adapted from Higham, http://personal.strath.ac.uk/d.j.higham/mil.m

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(101)
T=1; Delta=2**(-9); delta=Delta**2
L=int(T/Delta); K=int(Delta/delta)

X1 = np.zeros(L+1); X2 = np.zeros(L+1); X3 = np.zeros(L+1)
Y2 = 0

X1[0]=1; X2[0]=0.1; X3[0]=0.1
for j in range(1,L+1):
    Y1=0; Winc1=0; Winc2=0
    for k in range(1,K+1):
        dW1 = np.sqrt(delta)*np.random.randn(1)
        dW2 = np.sqrt(delta)*np.random.randn(1)
        Y1 += Y2*dW1
        Y2 += dW2
        Winc1 += dW1
        Winc2 += dW2
    X1[j] = X1[j-1] + X1[j-1]*X2[j-1]*Winc1 + \
            X1[j-1]*(X2[j-1]**2)*0.5*(Winc1**2 - Delta) + \
            0.3*X1[j-1]*X2[j-1]*Y1
    X2[j] = X2[j-1] - (X2[j-1] - X3[j-1])*Delta + 0.3*X2[j-1]*Winc2 + \
            0.9*X2[j-1]*0.5*(X2[j-1] - X3[j-1])*Delta
    X3[j] = X3[j-1] + (X2[j-1] - X3[j-1])*Delta

plt.plot(np.linspace(0,T,L+1), X1, 'r-')
plt.plot(np.linspace(0,T,L+1), X2, 'b--')
plt.plot(np.linspace(0,T,L+1), X3, 'k-.')
plt.legend(("$X^1$", "$X^2$", "$X^3$"))
plt.xlabel('t'); plt.ylabel('X',rotation=0)
plt.show()