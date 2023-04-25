#import necessary modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#define function for potential and background equation for inflaton field \phi
def potential(phi,potparams):
	a = potparams[0]
	b = potparams[1]
	c = potparams[2]
	v = a*phi**2 + b*np.sin(phi/c)
	dvdphi = 2*a*phi + (b/c)*np.cos(phi/c)
	return v,dvdphi

def bgeqn(Phi,N,potparams):
	phi = Phi[0]
	dphidN = Phi[1]
	v,dvdphi = potential(phi,potparams)
	d2phidN2 = -(3.-0.5*(dphidN*dphidN))*dphidN-(6.-(dphidN*dphidN))*dvdphi/(2.*v)
	return dphidN,d2phidN2

#parameter values
potparams = np.zeros(3)
potparams[0] =0.0015     #value of a
potparams[1] =0.00014     #value of b
potparams[2] =1/10    #value of c

#Initial conditions
phi0 = np.zeros(2)
phi0[0] = 21.45
Vini,dV_ini = potential(phi0[0],potparams)
phi0[1] = -dV_ini/Vini

#Number of efolds 
N = np.arange(0,120,5e-3)

#finding $\phi$ = phi[:,0] and $d\phi/dN$ = phi[:,1] using "odeint"
phi = odeint(bgeqn,phi0,N,args = (potparams,))

plt.figure(figsize=(12,9))
plt.title("$\phi$ vs efolds plot")
plt.plot(N,phi[:,0])
plt.xlabel("N")
plt.ylabel("$\phi$ in Mpl units")
plt.savefig('/home/barnali/Documents/GitHub/Cosmology/plots/phi2sin_phi.pdf')
plt.show()

plt.figure(figsize=(12,9))
plt.title("Derivative of $\phi$ vs efolds plot")
plt.plot(N,phi[:,1])
plt.xlabel("N")
plt.ylabel("d$\phi$/dN in Mpl units")
plt.savefig('/home/barnali/Documents/GitHub/Cosmology/plots/phi2sin_dphi.pdf')
plt.show()

plt.figure(figsize=(12,9))
plt.title("Phase space diagram of $\phi$")
plt.plot(phi[:,0],phi[:,1])
#plt.xlim(-1,1)
plt.xlabel("$\phi$ in Mpl units")
plt.ylabel("d$\phi$/dN in Mpl units")
plt.savefig('/home/barnali/Documents/GitHub/Cosmology/plots/phi2sin_Phasespace.pdf')
plt.show()

plt.figure(figsize=(12,9))
plt.title("V($\phi$) and dV($\phi$) vs $\phi$ plot")
V,dV = potential(phi[:,0],potparams)
plt.plot(phi[:,0],V,label = "Potential")
plt.plot(phi[:,0],dV,label = "Derivative of Potential")
plt.xlabel("$\phi$ in Mpl units")
plt.ylabel("V($\phi$) and dV($\phi$)")
plt.legend()
plt.savefig('/home/barnali/Documents/GitHub/Cosmology/plots/phi2sin_V_dV.pdf')
plt.show()

plt.figure(figsize=(12,9))
plt.title("Hubble parameter vs efolds")
H = np.sqrt(V/(3-((phi[:,1])**2)/2))
plt.plot(N,H)
plt.yscale('log')
plt.xlabel("N")
plt.ylabel("H(N)")
plt.savefig('/home/barnali/Documents/GitHub/Cosmology/plots/phi2sin_Hubbleparameter.pdf')
plt.show()

plt.figure(figsize=(12,9))
plt.title("Horizon vs efolds")
plt.plot(N,1/H)
plt.yscale('log')
plt.xlabel("N")
plt.ylabel("1/H(N)")
plt.savefig('/home/barnali/Documents/GitHub/Cosmology/plots/phi2sin_HubbleRadius.pdf')
plt.show()
