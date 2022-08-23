import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def potential(phi,potparams):
	v0 = potparams
	v = v0*(1 - (np.exp(-np.sqrt(2/3)*phi)))**2
	dvdphi = (2*v0*np.sqrt(2/3))*(1 - (np.exp(-np.sqrt(2/3)*phi)))*(np.exp(-np.sqrt(2/3)*phi))
	return v,dvdphi

def bgeqn(Phi,N,potparams):
	phi = Phi[0]
	dphidN = Phi[1]
	v,dvdphi = potential(phi,potparams)
	d2phidN2 = -(3.-0.5*(dphidN*dphidN))*dphidN-(6.-(dphidN*dphidN))*dvdphi/(2.*v)
	return dphidN,d2phidN2
	
potparams = (7e-6)**2

#Initial condition
phi0 = np.zeros(2)
phi0[0] = 5.62
Vini,dV_ini = potential(phi0[0],potparams)
phi0[1] = -dV_ini/Vini

N = np.arange(0,71,5e-3)

phi = odeint(bgeqn,phi0,N,args = (potparams,))

plt.plot(N,phi[:,0])
plt.xlabel("N")
plt.ylabel("phi in Mpl units")
plt.show()

plt.plot(N,phi[:,1])
plt.xlabel("N")
plt.ylabel("dphi/dN in Mpl units")
plt.show()

plt.plot(phi[:,0],phi[:,1])
#plt.xlim(-1,1)
plt.xlabel("phi")
plt.ylabel("dphi/dN in Mpl units")
plt.show()

V,dV = potential(phi[:,0],potparams)
#plt.yscale("log")
plt.plot(phi[:,0],V)
plt.plot(phi[:,0],dV)
plt.xlabel("phi in Mpl units")
plt.ylabel("V(Phi) and dV(phi)")
plt.legend("Potential","Derivative of Potential")
plt.show()

H = np.sqrt(V/(3-((phi[:,1])**2)/2))
plt.plot(N,H)
plt.yscale('log')
plt.xlabel("N")
plt.ylabel("H(N)")
plt.show()

plt.plot(N,1/H)
plt.yscale('log')
plt.xlabel("N")
plt.ylabel("1/H(N)")
plt.show()
