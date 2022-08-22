import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def potential(phi,potparams):
#	for m2phi2 potential
	mass = potparams
	v = 0.5*(mass**2)*phi*phi
	dvdphi = mass**2*phi
	
#	for lphi4 potential
#	Lambda = potparams
#	v = 0.25*Lambda*phi**4
#	dvdphi = Lambda*phi**3

#	for phi2sin potential
#	a = potparams[0]
#	b = potparams[1]
#	c = potparams[2]
#	v = a*phi**2 + b*np.sin(phi/c)
#	dvdphi = 2*a*phi + (b/c)*np.cos(phi/c)

#	for Punctuated Inflation potential
#	m = potparams[0]
#	a = potparams[1]
#	v = 0.5*(m*phi)**2 - (2/3)*(m*a)**2*(phi/a)**3 + 0.25*(m*a)**2*(phi/a)**4
#	dvdphi = m**2*phi - 2*m**2*a*(phi/a)**2 + m**2*a*(phi/a)**3
	
	return v,dvdphi

def bgeqn(Phi,N,potparams):
	phi = Phi[0]
	dphidN = Phi[1]
	v,dvdphi = potential(phi,potparams)
	d2phidN2 = -(3.-0.5*(dphidN*dphidN))*dphidN-(6.-(dphidN*dphidN))*dvdphi/(2.*v)
	return dphidN,d2phidN2

#for m2phi2 potential
potparams = 7e-6

#for lphi4 potential
#potparams = 7e-12

#for phi2sin potential
#potparams = np.zeros(3)
#potparams[0] =0.0015*1e-5     #value of a
#potparams[1] =0.00014*1e-5      #value of b
#potparams[2] =1/10    #value of c

#for Punctuated Inflation potential
#potparams = np.zeros(2)
#potparams[0] = 7.17*10**(-8)     #value of mass
#potparams[1] = 1.9654     #value of phi0(in paper) or a(in here) 

#Initial condition
phi0 = np.zeros(2)

#for m2phi2 potential
phi0[0] = 16.5

#for lphi4 potential
#phi0[0] = 23.65

#for phi2sin potential
#phi0[0] = 16.8

#for Punctuated Inflation potential
#phi0[0] = 14.3

Vini,dV_ini = potential(phi0[0],potparams)
phi0[1] = -dV_ini/Vini

N = np.arange(0,71,5e-3)  #put 120 instead of 71 for Punctuated inflation potential

phi = odeint(bgeqn,phi0,N,args = (potparams,))
V,dV = potential(phi[:,0],potparams)
H = np.sqrt(V/(3-((phi[:,1])**2)/2))
Rinfl = 1/H

def H(a,Or,Om,Ok,Ol,H0):
	Hz = H0*((Or/a**4)+(Om/a**3)+(Ok/a**2)+Ol)**0.50e0
	return Hz

Or=2.47e-5/(67.36/100.0)**2
Om=0.3153
Ok = 0.0e0
Ol=1-Or-Om-Ok
H0=67.36

C=2.99792458e8
Km=1000
MPc=3.085678e22
mGeVinv=5.07e15
MplGeV=2.4e18
Const=Km/MPc/C/mGeVinv/MplGeV

a = np.logspace(-71, 0, 1000)  #put 120 instead of 71 for Punctuated inflation potential
Hz=np.zeros(len(a))
Hz = H(a,Or,Om,Ok,Ol,H0)*Const
RadH = 1/Hz

lambda1=10**60*a
lambda2=10**57*a
lambda3=10**54*a

down = []
for i in lambda1:
	if i < Rinfl[0]:
		down.append(i)

tr = np.where(lambda1 == down[-1])[0][0]
pr = np.log10(a)[tr]
#for m2phi2 potential
extra = -2.75

#for lphi4 potential
#extra = -2.4

#for phi2sin potential
#extra = -2.9

#for Punctuated Inflation potential
#extra = -24

aN = N*0.4343+pr+extra

down1 = []
for i in RadH:
	if i < Rinfl[-1]:
		down1.append(i)

tr1 = np.where(RadH == down1[-1])[0][0]
pr1 = np.log10(a)[tr1]

plt.xlim([-60,0])
plt.xlabel(r'$\log_{10} a$')
plt.yscale('log')
plt.ylabel(r'$\frac{1}{H(a)} [1/M_{Pl}]$')
plt.plot(aN,Rinfl)
plt.plot(np.log10(a[tr1:]), RadH[tr1:])
plt.plot(np.log10(a),lambda1, lw=2,linestyle='--',label=r'$\lambda_1$')
plt.plot(np.log10(a),lambda2, lw=2,linestyle='--',label=r'$\lambda_2$')
plt.plot(np.log10(a),lambda3, lw=2,linestyle='--',label=r'$\lambda_3$')
plt.axvline(np.log10(1/1101), 0, 1, label='CMB',color='red')
plt.axvline(pr1, 0, 1, label='End of Inflation',color='violet')
plt.title('Physical lengthscales')
plt.legend()
plt.show()

plt.xlim([-60,0])
plt.xlabel(r'$\log_{10} a$')
plt.yscale('log')
plt.ylabel(r'$\frac{1}{aH(a)} [1/M_{Pl}]$')
plt.plot(aN,Rinfl/(10**aN))
plt.plot(np.log10(a[tr1:]),(RadH/a)[tr1:])
plt.plot(np.log10(a),lambda1/a, lw=2,linestyle='--',label=r'$\lambda_1$')
plt.plot(np.log10(a),lambda2/a, lw=2,linestyle='--',label=r'$\lambda_2$')
plt.plot(np.log10(a),lambda3/a, lw=2,linestyle='--',label=r'$\lambda_3$')
plt.axvline(np.log10(1/1101), 0, 1, label='CMB',color='red')
plt.axvline(pr1, 0, 1, label='End of Inflation',color='violet')
plt.title('Comoving lengthscales')
plt.legend()
plt.show()
