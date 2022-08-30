import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def potential(phi,potparams):
	m = potparams[0]
	a = potparams[1]
	v = 0.5*(m*phi)**2 - (2/3)*(m*a)**2*(phi/a)**3 + 0.25*(m*a)**2*(phi/a)**4
	dvdphi = m**2*phi - 2*m**2*a*(phi/a)**2 + m**2*a*(phi/a)**3
	return v,dvdphi

def bgeqn(Phi,N,potparams):
	phi = Phi[0]
	dphidN = Phi[1]
	v,dvdphi = potential(phi,potparams)
	d2phidN2 = -(3.-0.5*(dphidN*dphidN))*dphidN-(6.-(dphidN*dphidN))*dvdphi/(2.*v)
	return dphidN,d2phidN2
	
potparams = np.zeros(2)
potparams[0] = 7.17*10**(-8)     #value of mass
potparams[1] = 1.9654     #value of phi0(in paper) or a(in here) 

#Initial condition
phi0 = np.zeros(2)
phi0[0] = 12
Vini,dV_ini = potential(phi0[0],potparams)
phi0[1] = -dV_ini/Vini

N = np.arange(0,120,5e-3)

phi = odeint(bgeqn,phi0,N,args = (potparams,))

plt.title("$\phi$ vs efolds plot")
plt.plot(N,phi[:,0])
plt.xlabel("N")
plt.ylabel("$\phi$ in Mpl units")
plt.show()

plt.title("Derivative of $\phi$ vs efolds plot")
plt.plot(N,phi[:,1])
plt.xlabel("N")
plt.ylabel("d$\phi$/dN in Mpl units")
plt.show()

plt.title("Phase space diagram of $\phi$")
plt.plot(phi[:,0],phi[:,1])
#plt.xlim(-1,1)
plt.xlabel("$\phi$ in Mpl units")
plt.ylabel("d$\phi$/dN in Mpl units")
plt.show()

plt.title("V($\phi$) and dV($\phi$) vs $\phi$ plot")
V,dV = potential(phi[:,0],potparams)
plt.plot(phi[:,0],V,label = "Potential")
plt.plot(phi[:,0],dV,label = "Derivative of Potential")
plt.xlabel("$\phi$ in Mpl units")
plt.ylabel("V($\phi$) and dV($\phi$)")
plt.legend()
plt.show()

plt.title("Hubble parameter vs efolds")
H = np.sqrt(V/(3-((phi[:,1])**2)/2))
plt.plot(N,H)
plt.yscale('log')
plt.xlabel("N")
plt.ylabel("H(N)")
plt.show()

plt.title("Horizon vs efolds")
plt.plot(N,1/H)
plt.yscale('log')
plt.xlabel("N")
plt.ylabel("1/H(N)")
plt.show()
