import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def potential(phi,potparams):
	v0 = potparams[0]
	Ap = potparams[1]
	Am = potparams[2]
	phi00 = potparams[3]
	delphi = potparams[4]
	v = v0 + 0.5*(Ap+Am)*(phi-phi00) + 0.5*(Ap-Am)*(phi-phi00)*np.tanh((phi-phi00)/delphi)
	dvdphi = 0.5*(Ap+Am) + 0.5*(Ap-Am)*np.tanh((phi-phi00)/delphi) + 0.5*(Ap-Am)*(phi-phi00)*(1/(np.cosh((phi-phi00)/delphi)))**2/delphi
	return v,dvdphi

def bgeqn(Phi,N,potparams):
	phi = Phi[0]
	dphidN = Phi[1]
	v,dvdphi = potential(phi,potparams)
	d2phidN2 = -(3.-0.5*(dphidN*dphidN))*dphidN-(6.-(dphidN*dphidN))*dvdphi/(2.*v)
	return dphidN,d2phidN2
	
potparams = np.zeros(5)
potparams[0] = 5.3e-10 #2.37e-12    #value of v0
potparams[1] = 1e-1*potparams[0] #3.35e-14     #value of Ap 
potparams[2] = 1e-2*potparams[0] #7.26e-15    #value of Am
potparams[3] = 0.707     #value of phi00
potparams[4] = 2.6e-9     #value of delphi


#Initial condition
phi0 = np.zeros(2)
phi0[0] = 2.5e0
Vini,dV_ini = potential(phi0[0],potparams)
phi0[1] = -dV_ini/Vini

N = np.arange(0,71,5e-3)

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
