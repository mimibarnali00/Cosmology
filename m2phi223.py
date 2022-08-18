import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

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
#epsilon2
deps2dN = []
for i in range(np.size(N)-1):
	ans = ((epsilon1[i+1]-epsilon1[i])/(N[i+1]-N[i]))
	deps2dN.append(ans)

epsilon2 = np.divide(deps2dN,epsilon1[1:])

#Making size of epsilon1 = size of epsilon2 by adding a 0 as first element
zer = np.zeros(np.size(epsilon2)+1)
zer[1:] = epsilon2
epsilon2 = zer

#finding ai (Kp exits H at Ne=50)
kp = 0.05 #Mpc^-1
Np = N[infl-1]-50 #=Nend-50
ex = []
for i in N:
	if i < Np:
		ex.append(i)

ind1 = np.where(N == ex[-1])[0][0]
#print(ind1,N[ind1])
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
z = anew*np.sqrt(2*epsilon1)
dz = anew*Hinf*(anew*np.sqrt(2*epsilon1) + (epsilon1*epsilon2)/(np.sqrt(2*epsilon1)))

##initial conditions for scalars
#Gkrin = ((np.cos(kp*eta)/(np.sqrt(2*kp)))/z)[k_aHinind]
#dGkrin = (((-kp)*np.sin(kp*eta)/(np.sqrt(2*kp)))/z - (np.cos(kp*eta)/(np.sqrt(2*kp)))*dz/(z**2))[k_aHinind]
#Gkiin = ((-np.sin(kp*eta)/(np.sqrt(2*kp)))/z)[k_aHinind]
#dGkiin = (((-kp)*np.cos(kp*eta)/(np.sqrt(2*kp)))/z - (-np.sin(kp*eta)/(np.sqrt(2*kp)))*dz/(z**2))[k_aHinind]

#initial conditions
Gkrin = ((1/(np.sqrt(2*kp)))/z)[k_aHinind]
dGkrin = ((((-1/(2*np.sqrt(2*kp)))/z**2))*(dz+(2*z)))[k_aHinind]
Gkiin = 0
dGkiin = (-np.sqrt(kp/2)/(anew*Hinf*z))[k_aHinind]

##initial conditions for tensors
#hkrin = ((np.cos(kp*eta)/(np.sqrt(2*kp)))/anew)[k_aHinind]
#dhkrin = (((-kp)*np.sin(kp*eta)/(np.sqrt(2*kp)))/anew - (np.cos(kp*eta)/(np.sqrt(2*kp)))/anew)[k_aHinind]
#hkiin = ((-np.sin(kp*eta)/(np.sqrt(2*kp)))/anew)[k_aHinind]
#dhkiin = (((-kp)*np.cos(kp*eta)/(np.sqrt(2*kp)))/anew - (-np.sin(kp*eta)/(np.sqrt(2*kp)))/anew)[k_aHinind]

#initial conditions for tensors
hkrin = ((1/(np.sqrt(2*kp)))/anew)[k_aHinind]
dhkrin = ((-1/(np.sqrt(2*kp)))/anew)[k_aHinind]
hkiin = 0
dhkiin = -np.sqrt(kp/2)/(anew*Hinf*anew)[k_aHinind]

##Interpolating Hinf, epsilon1, epsilon2
Hinf_cubic   = interp1d(N, Hinf, kind='cubic')
epsilon1_cubic   = interp1d(N, epsilon1, kind='cubic')
epsilon2_cubic   = interp1d(N, epsilon2, kind='cubic')

#Perturbation eq in efolds
#G = psi + delphi/dphidN
def scalarperturbeq(N,G,k):
	a = ai*np.exp(N)
	Gk, GkN = G
	GkNN = - (3.0 - epsilon1_cubic(N) + epsilon2_cubic(N))*GkN - ((k/(a*Hinf_cubic(N)))**2)*Gk
	return GkN,GkNN

efolds = np.arange(Ni, Ne, 0.0001)

Nrfin = []
Nifin = []
Gr = []
dGr = []
Gi = []
dGi = []

Grsol = solve_ivp(scalarperturbeq,[Ni,Ne],[Gkrin,dGkrin],t_eval=efolds,args = (0.05, )) #default RK45
#print(Grsol)
Nrfin = Grsol.t
Gr = Grsol.y[0]
dGr = Grsol.y[1]

#############################################
Gisol = solve_ivp(scalarperturbeq,[Ni,Ne],[Gkiin,dGkiin],t_eval=efolds,args = (0.05, )) #default RK45
#print(Gisol)
Nifin = Gisol.t
Gi = Gisol.y[0]
dGi = Gisol.y[1]

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

ax1.plot(Nrfin,Gr,label = "Real")
ax1.plot(Nifin,Gi,label = "Imaginary")
ax1.set_xlabel('N')
ax1.set_ylabel('$\zeta_{k}$')
ax1.legend()

ax2.plot(Nrfin,np.abs(Gr),label = "Real")
ax2.plot(Nifin,np.abs(Gi),label = "Imaginary")
ax2.set_yscale('log')
ax2.set_xlabel('N')
ax2.set_ylabel('$\zeta_{k}$')
ax2.legend()

ax3.plot(Nrfin,dGr,label = "Real")
ax3.plot(Nifin,dGi,label = "Imaginary")
ax3.set_xlabel('N')
ax3.set_ylabel('$\zeta\'_{k}$')
ax3.legend()

ax4.plot(Nrfin,np.abs(dGr),label = "Real")
ax4.plot(Nifin,np.abs(dGi),label = "Imaginary")
ax4.set_yscale('log')
ax4.set_xlabel('N')
ax4.set_ylabel('$\zeta\'_{k}$')
ax4.legend()
plt.show()

#Tensor Perturbation eq in efolds
def tensorperturbeq(N,h,k):
	a = ai*np.exp(N)
	hk, hkN = h
	hkNN = - (3.0 - epsilon1_cubic(N))*hkN - ((k/(a*Hinf_cubic(N)))**2)*hk
	return hkN,hkNN

efolds = np.arange(Ni, Ne, 0.0001)

Nhrfin = []
Nhifin = []
hr = []
dhr = []
hi = []
dhi = []

hrsol = solve_ivp(tensorperturbeq,[Ni,Ne],[hkrin,dhkrin],t_eval=efolds,args = (0.05, )) #default RK45
#print(hrsol)
Nhrfin = hrsol.t
hr = hrsol.y[0]
dhr = hrsol.y[1]

#############################################
hisol = solve_ivp(tensorperturbeq,[Ni,Ne],[hkiin,dhkiin],t_eval=efolds,args = (0.05, )) #default RK45
#print(hisol)
Nhifin = hisol.t
hi = hisol.y[0]
dhi = hisol.y[1]

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

ax1.plot(Nhrfin,hr,label = "Real")
ax1.plot(Nhifin,hi,label = "Imaginary")
ax1.set_xlabel('N')
ax1.set_ylabel('$h_{k}$')
ax1.legend()

ax2.plot(Nhrfin,np.abs(hr),label = "Real")
ax2.plot(Nhifin,np.abs(hi),label = "Imaginary")
ax2.set_yscale('log')
ax2.set_xlabel('N')
ax2.set_ylabel('$h_{k}$')
ax2.legend()

ax3.plot(Nhrfin,dhr,label = "Real")
ax3.plot(Nhifin,dhi,label = "Imaginary")
ax3.set_xlabel('N')
ax3.set_ylabel('$h\'_{k}$')
ax3.legend()

ax4.plot(Nhrfin,np.abs(dhr),label = "Real")
ax4.plot(Nhifin,np.abs(dhi),label = "Imaginary")
ax4.set_yscale('log')
ax4.set_xlabel('N')
ax4.set_ylabel('$h\'_{k}$')
ax4.legend()
plt.show()

#scalar power spectrum 
def Ps(kk,aGkr,aGki):
	aGkr = np.array(aGkr)
	aGki = np.array(aGki)
	Ps = ((kk**3)/(2*np.pi**2))*((aGkr*aGkr)+(aGki*aGki)) #for a given k
	return Ps

k = np.logspace(-4, 0, 100)
finGr = []
finGi = []
for i in k:
	k_aH = i/(anew*Hinf)

	#initial N (Ni)
	k_aHin = []
	for j in k_aH:
		if j > 100:
			k_aHin.append(j)

	k_aHinind = np.where(k_aH == k_aHin[-1])[0][0]
	Ni = N[k_aHinind]
	
	#end N (Ne)
	k_aHen = []
	for j in k_aH:
		if j < 1e-5:
			k_aHen.append(j)

	k_aHenind = np.where(k_aH == k_aHen[0])[0][0]
	Ne = N[k_aHenind]
	efoldsg = np.arange(Ni, Ne, 0.0001)
	Gkrin = ((1/(np.sqrt(2*i)))/z)[k_aHinind]
	dGkrin = ((((-1/(2*np.sqrt(2*i)))/z**2))*(dz+(2*z)))[k_aHinind]
	Gkiin = 0
	dGkiin = (-np.sqrt(i/2)/(anew*Hinf*z))[k_aHinind]
	finGrsol = solve_ivp(scalarperturbeq,[Ni,Ne],[Gkrin,dGkrin],t_eval=efoldsg,args = (i, ))
	finGisol = solve_ivp(scalarperturbeq,[Ni,Ne],[Gkiin,dGkiin],t_eval=efoldsg,args = (i, ))
	finGr.append(finGrsol.y[0][-1])
	finGi.append(finGisol.y[0][-1])

#tensor power spectrum 
def Pt(kk,ahkr,ahki):
	ahkr = np.array(ahkr)
	ahki = np.array(ahki)
	Pt = ((8*kk**3)/(2*np.pi**2))*((ahkr*ahkr)+(ahki*ahki)) #for a given k
	return Pt

finhr = []
finhi = []
for i in k:
	k_aH = i/(anew*Hinf)

	#initial N (Ni)
	k_aHin = []
	for j in k_aH:
		if j > 100:
			k_aHin.append(j)

	k_aHinind = np.where(k_aH == k_aHin[-1])[0][0]
	Ni = N[k_aHinind]
	
	#end N (Ne)
	k_aHen = []
	for j in k_aH:
		if j < 1e-5:
			k_aHen.append(j)

	k_aHenind = np.where(k_aH == k_aHen[0])[0][0]
	Ne = N[k_aHenind]
	efoldsh = np.arange(Ni, Ne, 0.0001)
	
	hkrin = ((1/(np.sqrt(2*i)))/anew)[k_aHinind]
	dhkrin = ((-1/(np.sqrt(2*i)))/anew)[k_aHinind]
	hkiin = 0
	dhkiin = -np.sqrt(i/2)/(anew*Hinf*anew)[k_aHinind]
	finhrsol = solve_ivp(tensorperturbeq,[Ni,Ne],[hkrin,dhkrin],t_eval=efoldsh,args = (i, ))
	finhisol = solve_ivp(tensorperturbeq,[Ni,Ne],[hkiin,dhkiin],t_eval=efoldsh,args = (i, ))
	finhr.append(finhrsol.y[0][-1])
	finhi.append(finhisol.y[0][-1])

plt.plot(k,Ps(k,finGr,finGi))
plt.plot(k,Pt(k,finhr,finhi))
plt.xscale('log')
plt.xlabel("$k$ in $Mpc^{-1}$")
plt.ylabel("${\cal P}_{S/T}(k)$")
plt.show()

####writing values in files
np.savetxt('Background.txt', np.array([N, phi[:,0], epsilon1, V, Hinf, z, dz, phi[:,1], epsilon2, eta]).T, delimiter='\t', fmt="%s",header='N   phi   eps1   V   H   z   zN   phiN   eps2   eta')

np.savetxt('Perturbed.txt', np.array([efolds, Gr, Gi, dGr, dGi, hr, hi, dhr, dhi]).T, delimiter='\t', fmt="%s",header='N   realG   imgG   realGN   imgGN   realh   imgh   realhN   imghN')

np.savetxt('Powerspectrum.txt', np.array([k, finGr, finGi, Ps(k,finGr,finGi), finhr, finhi, Pt(k,finhr,finhi)]).T, delimiter='\t', fmt="%s",header='k   realGk   imgGk   Ps   realhk   imghk   Pt')

###by friday produce power spectrum
###double precision to remove fluctuations
