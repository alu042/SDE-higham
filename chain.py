#CHAIN Test stochastic chain rule
#
# Solve SDE for V(X) = sqrt(X) where X solves
#     dX = (alpha - X) dt + beta sqrt(X) dW,  X(0) = Xzero
#   with alpha = 2, beta = 1, Xzero = 1.
# Xem1 is Euler-Maruyama solution for X.
# Xem2 is Euler-Maruyama solution for SDE for V from chain rule.
# Hence, we compare sqrt(Xem1) and Xem2.

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(100)

alpha=2; beta=1; T=1; N=200; dt=float(T)/N
Xzero=1; Xzero2 = np.sqrt(Xzero)

Dt=dt
Xem1 = np.zeros(N); Xem2=np.zeros(N)
Xtemp1=Xzero; Xtemp2=Xzero
Xem1[0]=Xtemp1; Xem2[0]=Xtemp2

for j in xrange(1,int(N)):
    Winc=np.sqrt(dt)*np.random.randn(1)
    f1=(alpha-Xtemp1)
    g1=beta*np.sqrt(np.abs(Xtemp1))
    Xtemp1 += Dt*f1 + Winc*g1
    Xem1[j] = Xtemp1
    f2=(4.0*alpha-beta**2)/(8*Xtemp2) - Xtemp2/2.0
    g2=beta/2.0
    Xtemp2 += Dt*f2 + Winc*g2
    Xem2[j] = Xtemp2

ax=plt.subplot(111)
ax.plot(np.arange(0,T,Dt),np.sqrt(Xem1),'b-', np.arange(0,T,Dt),Xem2,'ro' )
ax.legend(("Direct solution", "Solution via chain rule"),loc=2)
plt.xlabel("t",fontsize=12)
plt.ylabel("V(X)",fontsize=12,rotation=0)

print "Max disrepancy is: ", np.max(np.abs(np.sqrt(Xem1)-Xem2))

plt.show()

