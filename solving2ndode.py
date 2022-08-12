import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

val = 10000
N = np.linspace(13,34,val)
Hinf = np.linspace(4,12,val)
epsilon1 = np.linspace(1,18,val)
epsilon2 = np.linspace(1,5,val)

def perturbeq(N,G,k,H,eps1,eps2):
	a = np.exp(N)
	Gk, GkN = G
	GkNN = - (3-eps1+eps2)*GkN - (k/(a*H))**2*Gk
	return GkN,GkNN

Gkin = 56
dGkin = 20
kp = 0.05

Gsol = solve_ivp(perturbeq,[N[0],N[1]],[Gkin,dGkin],args = (kp,Hinf[0],epsilon1[0],epsilon2[0])) #default is RK45

print(Gsol)

plt.plot(Gsol.t,Gsol.y[0])
plt.show()
plt.plot(Gsol.t,Gsol.y[1])
plt.show()

Gkin = Hinf*epsilon1
dGkin = Hinf*epsilon2

Nfin = []
G = []
dG = []
for i in range(np.size(N)-1):
	Gsol = solve_ivp(perturbeq,[N[i],N[i+1]],[Gkin[i],dGkin[i]],args = (kp,Hinf[i],epsilon1[i],epsilon2[i])) #default is RK45
	Nfin.append(Gsol.t[0])
	G.append(Gsol.y[0][0])
	dG.append(Gsol.y[1][0])
	
print(np.size(Nfin),np.size(G),np.size(dG))
plt.plot(Nfin,G)
plt.show()
