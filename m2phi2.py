import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

def potential(phi,potparams):
	mass = potparams
	v = 0.5*(mass**2)*phi*phi
	dvdphi = mass**2*phi
	return v,dvdphi

def bgeqn(Phi,N,potparams):
	phi = Phi[0]
	dphidN = Phi[1]
	v,dvdphi = potential(phi,potparams)
	d2phidN2 = -(3.-0.5*(dphidN*dphidN))*dphidN-(6.-(dphidN*dphidN))*dvdphi/(2.*v)
	return dphidN,d2phidN2
	
potparams = 7e-6

#Initial condition
phi0 = np.zeros(2)
phi0[0] = 16.5
Vini,dV_ini = potential(phi0[0],potparams)
phi0[1] = -dV_ini/Vini

N = np.arange(0,71,5e-3)

phi = odeint(bgeqn,phi0,N,args = (potparams,))
V,dV = potential(phi[:,0],potparams)
Hinf = np.sqrt(V/(3-((phi[:,1])**2)/2))
Rinfl = 1/Hinf

#Epsilon1
epsilon1 = 0.5*phi[:,1]*phi[:,1]
eps1 = [] #till end of inflation 
for i in epsilon1:
	eps1.append(i)
	if i > 1:
		break

infl = np.where(epsilon1 == eps1[-1])[0][0]
plt.ylabel(r'$\epsilon_1(N)$')
plt.xlabel(r'$N$')
plt.plot(N,epsilon1)
plt.show()

#print(epsilon1[infl-1])
plt.ylabel(r'$\epsilon_1(N)$')
plt.xlabel(r'$N$')
plt.plot(N[0:infl-1],epsilon1[0:infl-1])
plt.show()

def H(a,Or,Om,Ok,Ol,H0):
	Hz = H0*((Or/a**4)+(Om/a**3)+(Ok/a**2)+Ol)**0.50e0
	return Hz

Or=2.47e-5/(67.36/100.0)**2
Om=0.3153
Ok = 0.0e0
Ol=1-Or-Om-Ok
H0=67.36 #(km/s)/Mpc

C=2.99792458e8 #speed of light in m
Km=1000 #m
MPc=3.085678e22 #m
mGeVinv=5.07e15 #meter to GeV^-1
MplGeV=2.4e18 #Planck mass (1/sqrt(8 pi G)) to GeV

Const=Km/MPc/C/mGeVinv/MplGeV

a = np.logspace(-71, 0, 1000)
Hz = np.zeros(len(a))
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
extra = -2.75
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

##Perturbation equation
deps2dN = []
for i in range(np.size(N)-1):
	ans = ((epsilon1[i+1]-epsilon1[i])/(N[i+1]-N[i]))
	deps2dN.append(ans)

epsilon2 = np.divide(deps2dN,epsilon1[1:])
#Making size of epsilon1 = size of epsilon2 by adding a 0 as first element
zer = np.zeros(np.size(epsilon2)+1)
zer[1:] = epsilon2
epsilon2 = zer

#G = psi + delphi/dphidN
def perturbeq(N,G,k,H,eps1,eps2):
	a = np.exp(N)
	Gk, GkN = G
	GkNN = - (3-eps1+eps2)*GkN - (k/(a*H))**2*Gk
	return GkN,GkNN

#finding ai (Kp exits H at Ne=50)
kp = 0.05 #Mpc^-1
Np = N[infl-1]-50 #=Nend-50
ex = []
for i in N:
	if i < Np:
		ex.append(i)

ind1 = np.where(N == ex[-1])[0][0]
print(ind1,N[ind1])
ai = kp/(np.exp(N[ind1])*Hinf[ind1]) #Mpc^-1/Mpl
print("ai = ",ai)

anew = ai*np.exp(N)

k_aH = kp/(anew*Hinf)

#initial N (Ni)
k_aHin = []
for i in k_aH:
	if i > 100:
		k_aHin.append(i)

k_aHinind = np.where(k_aH == k_aHin[-1])[0][0]
Ni = N[k_aHinind]
print("k_aHin = ",k_aHin[-1])
print("Ni = ",Ni)

#end N (Ne)
k_aHen = []
for i in k_aH:
	if i < 1e-5:
		k_aHen.append(i)

k_aHenind = np.where(k_aH == k_aHen[0])[0][0]
Ne = N[k_aHenind]
print("k_aHen = ",k_aHen[0])
print("Ne = ",Ne)

#############
eta = -1/(anew*Hinf)

vk = np.exp(-i*kp*eta)/(np.sqrt(2*kp))
dvk = (-i*kp)*np.exp(-i*kp*eta)/(np.sqrt(2*kp))

z = anew*np.sqrt(2*epsilon1)
dz = anew*Hinf*(anew*np.sqrt(2*epsilon1) + (epsilon1*epsilon2)/(np.sqrt(2*epsilon1)))

Gk = vk/z
dGk = dvk/z - vk*dz/(z**2)

#initial conditions
ain = anew[k_aHinind]
etain = eta[k_aHinind]

vkin = vk[k_aHinind]
dvkin = dvk[k_aHinind]

zin = z[k_aHinind]
dzin = dz[k_aHinind]

Gkin = Gk[k_aHinind]
dGkin = dGk[k_aHinind]
#print(ain,etain,vkin,dvkin,zin,dzin,Gkin,dGkin)

N = np.linspace(Ni, Ne, np.size(Hinf))

Nfin = []
G = []
dG = []
for i in range(np.size(N)-1):
	Gsol = solve_ivp(perturbeq,[N[i],N[i+1]],[Gk[i],dGk[i]],args = (kp,Hinf[i],epsilon1[i],epsilon2[i]),dense_output=True) #default RK45
	Nfin.append(Gsol.t[0])
	G.append(Gsol.y[0][0])
	dG.append(Gsol.y[1][0])
	
print(np.size(Nfin),np.size(G),np.size(dG))
plt.plot(Nfin,G)
plt.yscale('log')
plt.xlabel('N')
plt.ylabel('$\zeta_{k}$')
plt.show()

plt.plot(Nfin,dG)
#plt.yscale('log')
plt.xlabel('N')
plt.ylabel('$\zeta\'_{k}$')
plt.show()
