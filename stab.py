#STAB Mean-square and asymptotic stability test for E-M
#
# SDE is  dX = gamma*X dt + mu*X dW,  X(0) = Xzero,
#      where gamma and mu are constants and Xzero = 1
#
# Adapted from 
# Desmond J. Higham "An Algorithmic Introduction to Numerical Simulation of 
#                    Stochastic Differential Equations"
#
# http://www.caam.rice.edu/~cox/stoch/dhigham.pdf


import numpy as np
import matplotlib.pyplot as plt
np.random.seed(100)

T=20; M=50000; Xzero=1

ax=plt.subplot(211)  ######## Mean Square #########
gamma=-3; mu=np.sqrt(3)
for k in range(1,4):
    Dt=2**(1-k)
    N=float(T)/Dt
    Xms=np.zeros((N,1)); Xtemp=Xzero*np.ones((M,1))
    Xms[0]=Xzero
    for j in range(1,int(N)):
        Winc=np.sqrt(Dt)*np.random.randn(M,1)
        Xtemp += Dt*gamma*Xtemp + mu*Xtemp*Winc
        Xms[j]=np.mean(Xtemp**2)
    ax.plot(np.arange(0,T,Dt), Xms)
    plt.yscale('log')

ax.legend(("\Delta t = 1", "\Delta t = 1/2", "\Delta t = 1/4"),loc=2)
plt.title("Mean square: \lambda = -3, \mu =  \sqrt{3}",fontsize=12)
plt.ylabel("E[X^2]",fontsize=10), plt.axis([0,T,1e-20,1e+20])

bx=plt.subplot(212)  ######## Asymptotic: a single path ##########
T=500
gamma=0.5; mu=np.sqrt(6)
for k in range(1,4):
    Dt=2**(1-k)
    N=float(T)/Dt
    Xemabs=np.zeros((N,1)); Xtemp=Xzero
    Xemabs[0]=Xzero
    for j in range(1,int(N)):
        Winc=np.sqrt(Dt)*np.random.randn(1)
        Xtemp += Dt*gamma*Xtemp + mu*Xtemp*Winc
        Xemabs[j]=np.abs(Xtemp)
    bx.plot(np.arange(0,T,Dt), Xemabs)
    plt.yscale('log')
bx.legend(("\Delta t = 1", "\Delta t = 1/2", "\Delta t = 1/4"),loc=2)
plt.title("Single path: \lambda = 1/2, \mu =  \sqrt{6}",fontsize=12)
plt.ylabel("|X|",fontsize=10), plt.axis([0,T,1e-50,1e+100])

plt.suptitle("Mean-square and asymptotic stability test for E-M",fontsize=16)
plt.show()
