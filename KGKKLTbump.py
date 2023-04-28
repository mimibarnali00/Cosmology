#import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib import rc, rcParams

######################################################################
rc('font', family='TimesRoman', weight = 'extra bold', size = 20.0)
rc('text', usetex=True)
rc('axes', linewidth = 2, labelsize = 'Large')  
rc('xtick', labelsize= 'large')
rcParams['xtick.major.size'] = 8.0 
rcParams['xtick.minor.size'] = 4.0
rcParams['xtick.major.pad'] = 8.0 
rcParams['xtick.minor.pad'] = 8.0
rc('ytick', labelsize= 'large')  
rcParams['ytick.major.size'] = 8.0 
rcParams['ytick.minor.size'] = 0.0
rcParams['ytick.major.pad'] = 3.0 
rcParams['ytick.minor.pad'] = 8.0
rc('lines', linewidth = 2, markeredgewidth=1.5)
rc('savefig', dpi=300)
######################################################################

#define function for potential and background equation for inflaton field \phi
def potential(phi,potparams):
#	v0 = potparams
	v0 = potparams[0]
	A = potparams[1]
	phi00 = potparams[2]
	s = potparams[3]
#	v = v0*phi**2/(0.5**2+phi**2)
#	dvdphi = v0*phi/(2*(0.5**2+phi**2)**2)
	v = v0*phi**2/(0.5**2+phi**2)*(1+A*np.exp(-0.5*(phi-phi00)**2/s**2))
	dvdphi = 2*v0*phi/(0.5**2+phi**2)*(1+A*np.exp(-0.5*(phi-phi00)**2/s**2))-2*v0*phi**3/(0.5**2+phi**2)**2*(1+A*np.exp(-0.5*(phi-phi00)**2/s**2))-A*v0*phi**2*(phi-phi00)*np.exp(-0.5*(phi-phi00)**2/s**2)/((0.5**2+phi**2)*s**2)
	return v,dvdphi

def bgeqn(Phi,N,potparams):
	phi = Phi[0]
	dphidN = Phi[1]
	v,dvdphi = potential(phi,potparams)
	d2phidN2 = -(3.-0.5*(dphidN*dphidN))*dphidN-(6.-(dphidN*dphidN))*dvdphi/(2.*v)
	return dphidN,d2phidN2

#parameter values
#potparams = (7e-6)**2
potparams = np.zeros(4)
potparams[0] = (7e-6)**2 #value of v0
potparams[1] = 1.876e-3 #value of A 
potparams[2] = 2.005 #value of phi00
potparams[3] = 1.993e-2 #value of s

#Initial conditions
phi0 = np.zeros(2)
phi0[0] = 3.01 #3.3 without bump
Vini,dV_ini = potential(phi0[0],potparams)
phi0[1] = -dV_ini/Vini

#Number of efolds 
N = np.arange(0,71,5e-3)

#finding $\phi$ = phi[:,0] and $d\phi/dN$ = phi[:,1] using "odeint"
phi = odeint(bgeqn,phi0,N,args = (potparams,))

plt.figure(figsize=(12,9))
plt.title("$\phi$ vs efolds plot")
plt.plot(phi[:,0],N)
plt.xlabel("$\phi$ in Mpl units")
plt.ylabel("N")
plt.show()

plt.figure(figsize=(12,9))
plt.title("$\phi$ vs efolds plot")
plt.plot(N,phi[:,0])
plt.xlabel("N")
plt.ylabel("$\phi$ in Mpl units")
plt.savefig('/home/barnali/Documents/GitHub/Cosmology/plots/KKLTbump_phi.pdf')
plt.show()

plt.figure(figsize=(12,9))
plt.title("Derivative of $\phi$ vs efolds plot")
plt.plot(N,phi[:,1])
plt.xlabel("N")
plt.ylabel("d$\phi$/dN in Mpl units")
plt.savefig('/home/barnali/Documents/GitHub/Cosmology/plots/KKLTbump_dphi.pdf')
plt.show()

plt.figure(figsize=(12,9))
plt.title("Phase space diagram of $\phi$")
plt.plot(phi[:,0],phi[:,1])
#plt.xlim(-1,1)
plt.xlabel("$\phi$ in Mpl units")
plt.ylabel("d$\phi$/dN in Mpl units")
plt.savefig('/home/barnali/Documents/GitHub/Cosmology/plots/KKLTbump_Phasespace.pdf')
plt.show()

plt.figure(figsize=(12,9))
plt.title("V($\phi$) and dV($\phi$) vs $\phi$ plot")
V,dV = potential(phi[:,0],potparams)
plt.plot(phi[:,0],V,label = "Potential")
plt.plot(phi[:,0],dV,label = "Derivative of Potential")
plt.xlabel("$\phi$ in Mpl units")
plt.ylabel("V($\phi$) and dV($\phi$)")
plt.legend()
plt.savefig('/home/barnali/Documents/GitHub/Cosmology/plots/KKLTbump_V_dV.pdf')
plt.show()

plt.figure(figsize=(12,9))
plt.title("Hubble parameter vs efolds")
H = np.sqrt(V/(3-((phi[:,1])**2)/2))
plt.plot(N,H)
plt.yscale('log')
plt.xlabel("N")
plt.ylabel("H(N)")
plt.savefig('/home/barnali/Documents/GitHub/Cosmology/plots/KKLTbump_Hubbleparameter.pdf')
plt.show()

plt.figure(figsize=(12,9))
plt.title("Horizon vs efolds")
plt.plot(N,1/H)
plt.yscale('log')
plt.xlabel("N")
plt.ylabel("1/H(N)")
plt.savefig('/home/barnali/Documents/GitHub/Cosmology/plots/KKLTbump_HubbleRadius.pdf')
plt.show()
