#import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
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

#parameter values
potparams = 7e-6

#Initial conditions
phi0 = np.zeros(2)
phi0[0] = 16.5
Vini,dV_ini = potential(phi0[0],potparams)
phi0[1] = -dV_ini/Vini

#Number of efolds 
N = np.arange(0,71,5e-3)

#finding $\phi$ = phi[:,0] and $d\phi/dN$ = phi[:,1] using "odeint"
phi = odeint(bgeqn,phi0,N,args = (potparams,))
V,dV = potential(phi[:,0],potparams)
Hinf = np.sqrt(V/(3-((phi[:,1])**2)/2))
Rinfl = 1/Hinf

#\epsilon_1
epsilon1 = 0.5*phi[:,1]*phi[:,1]

plt.figure(figsize=(12,9))
plt.ylabel(r'$\epsilon_1(N)$')
plt.xlabel(r'$N$')
plt.plot(N,epsilon1)
plt.title("$\epsilon_1$(N) vs N plot")
plt.show()

#\epsilon_1 till the end of inflation
eps1 = []
for i in epsilon1:
	eps1.append(i)
	if i > 1:
		break

infl = np.where(epsilon1 == eps1[-1])[0][0]

plt.figure(figsize=(12,9))
plt.ylabel(r'$\epsilon_1(N)$')
plt.xlabel(r'$N$')
plt.plot(N[0:infl-1],epsilon1[0:infl-1])
plt.title("$\epsilon_1$(N) vs N plot till the end of Inflation")
plt.show()

##Horizon for radiation, matter and dark energy
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

#physical wavelengths for reference
lambda1=10**60*a
lambda2=10**57*a
lambda3=10**54*a

#Matching the Hubble radius during inflation to that of radiation
down = []
for i in lambda1:
	if i < Rinfl[0]:
		down.append(i)

tr = np.where(lambda1 == down[-1])[0][0]
pr = np.log10(a)[tr]
extra = -2.75 #adding some value to alomst exactly match the Hubble radius during inflation to that of radiation for visualization purposes.
aN = N*0.4343+pr+extra

down1 = []
for i in RadH:
	if i < Rinfl[-1]:
		down1.append(i)

tr1 = np.where(RadH == down1[-1])[0][0]
pr1 = np.log10(a)[tr1]

plt.figure(figsize=(12,9))
plt.xlim([-60,0])
plt.xlabel(r'$\log_{10} a$')
plt.yscale('log')
plt.ylabel(r'$\frac{M_{_{\textrm{Pl}}}}{H(a)}$')
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

plt.figure(figsize=(12,9))
plt.xlim([-60,0])
plt.xlabel(r'$\log_{10} a$')
plt.yscale('log')
plt.ylabel(r'$\frac{M_{_{\textrm{Pl}}}}{aH(a)}$')
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
#\epsilon_2
deps2dN = []
for i in range(np.size(N)-1):
	ans = ((epsilon1[i+1]-epsilon1[i])/(N[i+1]-N[i]))
	deps2dN.append(ans)

epsilon2 = np.divide(deps2dN,epsilon1[1:])

#Making size of \epsilon_1 = size of \epsilon_2 by adding a 0 as first element
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

ai = kp/(np.exp(N[ind1])*Hinf[ind1]) #Mpc^-1/Mpl

#redefining quantities such that their last value corresponds to end of inflation (for numerical efficiency)
N = N[0:infl-1]
Hinf = Hinf[0:infl-1]
epsilon1 = epsilon1[0:infl-1]
epsilon2 = epsilon2[0:infl-1]
V = V[0:infl-1]
dV = dV[0:infl-1]

anew = ai*np.exp(N)

def NiNe(kkp):
	k_aH = kkp/(anew*Hinf)
	#initial N (Ni)
	k_aHinind = np.abs(k_aH - 100).argmin()
	k_aHin = k_aH[k_aHinind]
	Ni = N[k_aHinind]
	#end N (Ne)
	k_aHenind = np.abs(k_aH - 1e-5).argmin()
	k_aHen = k_aH[k_aHenind]
	Ne = N[k_aHenind]
	
	efolds = np.arange(Ni, Ne, 0.0001)
	if np.amax(efolds) > Ne:
		Ne = N[-1]
	
	efolds = np.arange(Ni, Ne, 0.0001)[:-1]
	return k_aHin,k_aHinind,Ni,k_aHen,k_aHenind,Ne,efolds

#defining conformal time (\eta), z, dz/d\eta
eta = -1/(anew*Hinf)[NiNe(kp)[1]]
#z = anew*np.sqrt(2*epsilon1)
#dz = anew*Hinf*(anew*np.sqrt(2*epsilon1) + (epsilon1*epsilon2)/(np.sqrt(2*epsilon1)))
d2phi=-(3.-0.5*(phi[0:infl-1,1]*phi[0:infl-1,1]))*phi[0:infl-1,1]-(6.-(phi[0:infl-1,1]*phi[0:infl-1,1]))*dV/(2.*V)
z = anew*phi[0:infl-1,1]
dz = anew*(phi[0:infl-1,1]+d2phi)

##initial conditions for scalars
Gkrin = ((np.cos(kp*eta)/(np.sqrt(2*kp)))/z)[NiNe(kp)[1]]
dGkrin = (((-kp)*np.sin(kp*eta)/(np.sqrt(2*kp)))/z - (np.cos(kp*eta)/(np.sqrt(2*kp)))*dz/(z**2))[NiNe(kp)[1]]
Gkiin = ((-np.sin(kp*eta)/(np.sqrt(2*kp)))/z)[NiNe(kp)[1]]
dGkiin = (((-kp)*np.cos(kp*eta)/(np.sqrt(2*kp)))/z - (-np.sin(kp*eta)/(np.sqrt(2*kp)))*dz/(z**2))[NiNe(kp)[1]]

#initial conditions for scalars with phase factor = 0
#Gkrin = ((1/(np.sqrt(2*kp)))/z)[NiNe(kp)[1]]
#dGkrin = ((((-1/(2*np.sqrt(2*kp)))/z**2))*(dz+(2*z)))[NiNe(kp)[1]]
#Gkiin = 0
#dGkiin = (-np.sqrt(kp/2)/(anew*Hinf*z))[NiNe(kp)[1]]

##initial conditions for tensors
hkrin = ((np.cos(kp*eta)/(np.sqrt(2*kp)))/anew)[NiNe(kp)[1]]
dhkrin = (((-kp)*np.sin(kp*eta)/(np.sqrt(2*kp)))/anew - (Hinf*np.cos(kp*eta)/(np.sqrt(2*kp))))[NiNe(kp)[1]]
hkiin = ((-np.sin(kp*eta)/(np.sqrt(2*kp)))/anew)[NiNe(kp)[1]]
dhkiin = (((-kp)*np.cos(kp*eta)/(np.sqrt(2*kp)))/anew + (Hinf*np.sin(kp*eta)/(np.sqrt(2*kp))))[NiNe(kp)[1]]

#initial conditions for tensors with phase factor = 0
#hkrin = ((1/(np.sqrt(2*kp)))/anew)[NiNe(kp)[1]]
#dhkrin = ((-1/(np.sqrt(2*kp)))/anew)[NiNe(kp)[1]]
#hkiin = 0
#dhkiin = -np.sqrt(kp/2)/(anew*Hinf*anew)[NiNe(kp)[1]]

##initial conditions for vectors
vkrin = ((np.cos(kp*eta)/(np.sqrt(2*kp)))/anew)[NiNe(kp)[1]]
dvkrin = (((-kp)*np.sin(kp*eta)/(np.sqrt(2*kp)))/anew - (Hinf*np.cos(kp*eta)/(np.sqrt(2*kp))))[NiNe(kp)[1]]
vkiin = ((-np.sin(kp*eta)/(np.sqrt(2*kp)))/anew)[NiNe(kp)[1]]
dvkiin = (((-kp)*np.cos(kp*eta)/(np.sqrt(2*kp)))/anew + (Hinf*np.sin(kp*eta)/(np.sqrt(2*kp))))[NiNe(kp)[1]]

##Interpolating Hinf, epsilon1, epsilon2
Hinf_cubic   = interp1d(N, Hinf, kind='cubic')
epsilon1_cubic   = interp1d(N, epsilon1, kind='cubic')
epsilon2_cubic   = interp1d(N, epsilon2, kind='cubic')

#Scalar Perturbation eq in efolds
#G = psi + delphi/dphidN
def scalarperturbeq(N,G,k):
	a = ai*np.exp(N)
	Gk, GkN = G
	GkNN = - (3.0 - epsilon1_cubic(N) + epsilon2_cubic(N))*GkN - ((k/(a*Hinf_cubic(N)))**2)*Gk
	return GkN,GkNN

#solving for real part
Nrfin = []
Gr = []
dGr = []

Grsol = solve_ivp(scalarperturbeq,[NiNe(kp)[2],NiNe(kp)[5]],[Gkrin,dGkrin],t_eval=NiNe(kp)[6],args = (kp, ),atol=1e-32) #default RK45

Nrfin = Grsol.t
Gr = Grsol.y[0]
dGr = Grsol.y[1]

#solving for imaginary part
Nifin = []
Gi = []
dGi = []

Gisol = solve_ivp(scalarperturbeq,[NiNe(kp)[2],NiNe(kp)[5]],[Gkiin,dGkiin],t_eval=NiNe(kp)[6],args = (kp, ),atol=1e-32) #default RK45

Nifin = Gisol.t
Gi = Gisol.y[0]
dGi = Gisol.y[1]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(12,9))
fig.suptitle("Mode evolution plot (Scalar) for pivot scale (k = 0.05 $M pc^{-1}$))")

ax1.plot(Nrfin,Gr,label = "Real")
ax1.plot(Nifin,Gi,label = "Imaginary")
ax1.set_xlabel('N')
ax1.set_ylabel('$\zeta_{k}$')
#ax1.legend()
ax1.legend(loc="upper right",frameon=False)

ax2.plot(Nrfin,np.abs(Gr),label = "Real")
ax2.plot(Nifin,np.abs(Gi),label = "Imaginary")
ax2.set_yscale('log')
ax2.set_xlabel('N')
ax2.set_ylabel('$\zeta_{k}$')
#ax2.legend()

ax3.plot(Nrfin,dGr,label = "Real")
ax3.plot(Nifin,dGi,label = "Imaginary")
ax3.set_xlabel('N')
ax3.set_ylabel('$\zeta\'_{k}$')
#ax3.legend()

ax4.plot(Nrfin,np.abs(dGr),label = "Real")
ax4.plot(Nifin,np.abs(dGi),label = "Imaginary")
ax4.set_yscale('log')
ax4.set_xlabel('N')
ax4.set_ylabel('$\zeta\'_{k}$')
#ax4.legend()
plt.subplots_adjust(left=0.15,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
plt.show()

#Tensor Perturbation eq in efolds
def tensorperturbeq(N,h,k):
	a = ai*np.exp(N)
	hk, hkN = h
	hkNN = - (3.0 - epsilon1_cubic(N))*hkN - ((k/(a*Hinf_cubic(N)))**2)*hk
	return hkN,hkNN

#solving for real part
Nhrfin = []
hr = []
dhr = []

hrsol = solve_ivp(tensorperturbeq,[NiNe(kp)[2],NiNe(kp)[5]],[hkrin,dhkrin],t_eval=NiNe(kp)[6],args = (kp, ),atol=1e-32) #default RK45

Nhrfin = hrsol.t
hr = hrsol.y[0]
dhr = hrsol.y[1]

#solving for imaginary part
Nhifin = []
hi = []
dhi = []

hisol = solve_ivp(tensorperturbeq,[NiNe(kp)[2],NiNe(kp)[5]],[hkiin,dhkiin],t_eval=NiNe(kp)[6],args = (kp, ),atol=1e-32) #default RK45

Nhifin = hisol.t
hi = hisol.y[0]
dhi = hisol.y[1]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(12,9))
fig.suptitle("Mode evolution plot (Tensor) for pivot scale (k = 0.05 $M pc^{-1}$))")

ax1.plot(Nhrfin,hr,label = "Real")
ax1.plot(Nhifin,hi,label = "Imaginary")
ax1.set_xlabel('N')
ax1.set_ylabel('$h_{k}$')
#ax1.legend()
ax1.legend(loc="upper right",frameon=False)

ax2.plot(Nhrfin,np.abs(hr),label = "Real")
ax2.plot(Nhifin,np.abs(hi),label = "Imaginary")
ax2.set_yscale('log')
ax2.set_xlabel('N')
ax2.set_ylabel('$h_{k}$')
#ax2.legend()

ax3.plot(Nhrfin,dhr,label = "Real")
ax3.plot(Nhifin,dhi,label = "Imaginary")
ax3.set_xlabel('N')
ax3.set_ylabel('$h\'_{k}$')
#ax3.legend()

ax4.plot(Nhrfin,np.abs(dhr),label = "Real")
ax4.plot(Nhifin,np.abs(dhi),label = "Imaginary")
ax4.set_yscale('log')
ax4.set_xlabel('N')
ax4.set_ylabel('$h\'_{k}$')
#ax4.legend()
plt.subplots_adjust(left=0.15,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
plt.show()

#Vector Perturbation eq in efolds
mm = 7e-8
def vectorperturbeq(N,v,k):
	a = ai*np.exp(N)
	vk, vkN = v
	vkNN = - (1.0 - epsilon1_cubic(N) + 2*(k**2)/(a**2*mm**2 + k**2))*vkN - ((a**2*mm**2 + k**2)/(a*Hinf_cubic(N))**2)*vk
	return vkN,vkNN

#solving for real part
Nvrfin = []
vr = []
dvr = []

vrsol = solve_ivp(vectorperturbeq,[NiNe(kp)[2],NiNe(kp)[5]],[vkrin,dvkrin],t_eval=NiNe(kp)[6],args = (kp, ),atol=1e-32) #default RK45

Nvrfin = vrsol.t
vr = vrsol.y[0]
dvr = vrsol.y[1]

#solving for imaginary part
Nvifin = []
vi = []
dvi = []

visol = solve_ivp(vectorperturbeq,[NiNe(kp)[2],NiNe(kp)[5]],[vkiin,dvkiin],t_eval=NiNe(kp)[6],args = (kp, ),atol=1e-32) #default RK45

Nvifin = visol.t
vi = visol.y[0]
dvi = visol.y[1]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(12,9))
fig.suptitle("Mode evolution plot (Vector) for pivot scale (k = 0.05 $M pc^{-1}$))")

ax1.plot(Nvrfin,vr,label = "Real")
ax1.plot(Nvifin,vi,label = "Imaginary")
ax1.set_xlabel('N')
ax1.set_ylabel('$v_{k}$')
#ax1.legend()
ax1.legend(loc="upper right",frameon=False)

ax2.plot(Nvrfin,np.abs(vr),label = "Real")
ax2.plot(Nvifin,np.abs(vi),label = "Imaginary")
ax2.set_yscale('log')
ax2.set_xlabel('N')
ax2.set_ylabel('$v_{k}$')
#ax2.legend()

ax3.plot(Nvrfin,dvr,label = "Real")
ax3.plot(Nvifin,dvi,label = "Imaginary")
ax3.set_xlabel('N')
ax3.set_ylabel('$v\'_{k}$')
#ax3.legend()

ax4.plot(Nvrfin,np.abs(dvr),label = "Real")
ax4.plot(Nvifin,np.abs(dvi),label = "Imaginary")
ax4.set_yscale('log')
ax4.set_xlabel('N')
ax4.set_ylabel('$v\'_{k}$')
#ax4.legend()
plt.subplots_adjust(left=0.15,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
plt.savefig('/home/barnali/Documents/GitHub/Cosmology/plots/vector_model_m2phi2_Vectorperturbation_mm7em8.pdf')
plt.show()

#scalar power spectrum 
def Ps(kk,aGkr,aGki):
	aGkr = np.array(aGkr)
	aGki = np.array(aGki)
	Ps = ((kk**3)/(2*np.pi**2))*((aGkr*aGkr)+(aGki*aGki)) #for a given k
	return Ps

#k = np.logspace(-4, 0, 100)
k = np.logspace(-3, 2, 100)
finGr = []
finGi = []
for i in k:
	Gkrin = ((1/(np.sqrt(2*i)))/z)[NiNe(i)[1]]
	dGkrin = ((((-1/(2*np.sqrt(2*i)))/z**2))*(dz+(2*z)))[NiNe(i)[1]]
	Gkiin = 0
	dGkiin = (-np.sqrt(i/2)/(anew*Hinf*z))[NiNe(i)[1]]
	finGrsol = solve_ivp(scalarperturbeq,[NiNe(i)[2],NiNe(i)[5]],[Gkrin,dGkrin],t_eval=NiNe(i)[6],args = (i, ),atol=1e-32)
	finGisol = solve_ivp(scalarperturbeq,[NiNe(i)[2],NiNe(i)[5]],[Gkiin,dGkiin],t_eval=NiNe(i)[6],args = (i, ),atol=1e-32)
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
	hkrin = ((1/(np.sqrt(2*i)))/anew)[NiNe(i)[1]]
	dhkrin = ((-1/(np.sqrt(2*i)))/anew)[NiNe(i)[1]]
	hkiin = 0
	dhkiin = -np.sqrt(i/2)/(anew*Hinf*anew)[NiNe(i)[1]]
	finhrsol = solve_ivp(tensorperturbeq,[NiNe(i)[2],NiNe(i)[5]],[hkrin,dhkrin],t_eval=NiNe(i)[6],args = (i, ),atol=1e-32)
	finhisol = solve_ivp(tensorperturbeq,[NiNe(i)[2],NiNe(i)[5]],[hkiin,dhkiin],t_eval=NiNe(i)[6],args = (i, ),atol=1e-32)
	finhr.append(finhrsol.y[0][-1])
	finhi.append(finhisol.y[0][-1])

#vector power spectrum
def Pv(kk,avkr,avki):
	avkr = np.array(avkr)
	avki = np.array(avki)
	Pv = ((4*kk**3)/(2*np.pi**2))*((avkr*avkr)+(avki*avki)) #for a given k
	return Pv

finvr = []
finvi = []
for i in k:
	B = np.sqrt(i**2 + anew**2*mm**2)
	vkrin = ((B*np.cos(B*eta)/(np.sqrt(2*i)))/(i*anew))[NiNe(i)[1]]
	dvkrin = (((-i**2/B)*np.cos(B*eta)/(np.sqrt(2*i)))/anew - (B**2*np.sin(B*eta)/(np.sqrt(2*i)))/(i*anew))[NiNe(i)[1]]
	vkiin = ((-B*np.sin(B*eta)/(np.sqrt(2*i)))/(i*anew))[NiNe(i)[1]]
	dvkiin = ((-B**2*np.cos(B*eta)/(np.sqrt(2*i)))/anew - (-(i**2/B)*np.sin(B*eta)/(np.sqrt(2*i)))/(i*anew))[NiNe(i)[1]]
	finvrsol = solve_ivp(vectorperturbeq,[NiNe(i)[2],NiNe(i)[5]],[vkrin,dvkrin],t_eval=NiNe(i)[6],args = (i, ),atol=1e-32)
	finvisol = solve_ivp(vectorperturbeq,[NiNe(i)[2],NiNe(i)[5]],[vkiin,dvkiin],t_eval=NiNe(i)[6],args = (i, ),atol=1e-32)
	finvr.append(finvrsol.y[0][-1])
	finvi.append(finvisol.y[0][-1])

#Analytic Vector Power Spectrum
def PvA(N,kk):
	a = ai*np.exp(N)
	kk_aH = kk/(a*Hinf_cubic(N))
	kaHindex = np.abs(kk_aH - 1).argmin()
	PvA = ((kk*Hinf_cubic(N[kaHindex]))/(2*np.pi*mm))**2 
	return PvA

print(PvA(NiNe(i)[6],10))

#Slow roll approximation
def PSR(N,kk):
	a = ai*np.exp(N)
	kk_aH = kk/(a*Hinf_cubic(N))
	kaHindex = np.abs(kk_aH - 1).argmin()
	
	Psk = (0.5/epsilon1_cubic(N[kaHindex]))*(Hinf_cubic(N[kaHindex])/(2*np.pi))**2
	Ptk = 8*(Hinf_cubic(N[kaHindex])/(2*np.pi))**2
	rk = 16*epsilon1_cubic(N[kaHindex])
	nsk = 1-2*epsilon1_cubic(N[kaHindex]) - epsilon2_cubic(N[kaHindex])
	return Psk,Ptk,rk,nsk

pvvA = []
for i in k:
	pvvA.append(PvA(NiNe(i)[6],i))

pssr = []
ptsr = []
rsr = []
nssr = []
for i in k:
	pssr.append(PSR(NiNe(i)[6],i)[0])
for i in k:
	ptsr.append(PSR(NiNe(i)[6],i)[1])
for i in k:
	rsr.append(PSR(NiNe(i)[6],i)[2])
for i in k:
	nssr.append(PSR(NiNe(i)[6],i)[3])

plt.figure(figsize=(12,9))
plt.plot(k/kp,Pv(k,finvr,finvi),label="Vector Power spectrum")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("$k/kp/Mpc^{-1}$")
plt.ylabel("${\cal P}_{V}(k)$")
plt.legend()
plt.savefig('/home/barnali/Documents/GitHub/Cosmology/plots/vector_model_m2phi2_VectorPowerspectrum_mm7em8.pdf')
plt.show()

plt.figure(figsize=(12,9))
plt.plot(k,Ps(k,finGr,finGi),label="Scalar Power spectrum")
plt.plot(k,Pt(k,finhr,finhi),label="Tensor Power spectrum")
plt.plot(k,Pv(k,finvr,finvi),label="Vector Power spectrum")
plt.plot(k,pssr,'g--',label="Scalar Power spectrum (Slow roll approximation)")
plt.plot(k,ptsr,'m--',label="Tensor Power spectrum (Slow roll approximation)")
plt.plot(k,pvvA,'k--',label="Vector Power spectrum (Analytic)")
plt.title("Primordial power spectra")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("$k/Mpc^{-1}$")
plt.ylabel("${\cal P}_{S/T/V}(k)$")
plt.legend()
plt.savefig('/home/barnali/Documents/GitHub/Cosmology/plots/vector_model_m2phi2_Powerspectra_mm7em8.pdf')
plt.show()

percenterrscalar = (np.abs(Ps(k,finGr,finGi)-pssr)/pssr)*100

plt.figure(figsize=(12,9))
plt.title("Percentage error in scalar power spectra")
plt.plot(k,percenterrscalar)
plt.xlabel("$k/Mpc^{-1}$")
plt.ylabel("$\%$ error")
plt.show()

percenterrtensor = (np.abs(Pt(k,finhr,finhi)-ptsr)/ptsr)*100

plt.figure(figsize=(12,9))
plt.title("Percentage error in tensor power spectra")
plt.plot(k,percenterrtensor)
plt.xlabel("$k/Mpc^{-1}$")
plt.ylabel("$\%$ error")
plt.show()

plt.figure(figsize=(12,9))
plt.plot(k,Pt(k,finhr,finhi)/Ps(k,finGr,finGi))
plt.plot(k,rsr,'g--',label="Slow roll approximation")
plt.title("Tensor to scalar ratio")
plt.xscale('log')
plt.xlabel("$k/Mpc^{-1}$")
plt.ylabel("r(k)")
#plt.ylim([0,0.5])
plt.show()

#spectral tilt [ns = 1+((d ln Ps)/(d ln k))]
lnPs = np.log(Ps(k,finGr,finGi))
lnk = np.log(k)
dlnk = lnk[1]-lnk[0]
ns = 1+np.gradient(lnPs,dlnk)

plt.figure(figsize=(12,9))
plt.plot(k[0:-1],ns[0:-1])
plt.plot(k[0:-1],nssr[0:-1],'g--',label="Slow roll approximation")
plt.title("Spectral index")
plt.xscale('log')
plt.xlabel("$k/Mpc^{-1}$")
plt.ylabel("$n_{s}(k)$")
plt.ylim([0.9,1])
plt.show()

#writing values in files (change the file destination accordingly)
np.savetxt('/home/barnali/Documents/GitHub/Cosmology/files/Backgroundm2phi2vec.txt', np.array([N, phi[0:infl-1,0], epsilon1, V, Hinf, z, dz, phi[0:infl-1,1], epsilon2]).T, delimiter='\t', fmt="%s",header='N    phi    eps1    V    H    z    zN    phiN    eps2')

np.savetxt('/home/barnali/Documents/GitHub/Cosmology/files/Perturbedm2phi2vec.txt', np.array([NiNe(kp)[6], Gr, Gi, dGr, dGi, hr, hi, dhr, dhi, vr, vi, dvr, dvi]).T, delimiter='\t', fmt="%s",header='N    realG    imgG    realGN    imgGN    realh    imgh    realhN    imghN    realv    imgv    realvN    imgvN')

np.savetxt('/home/barnali/Documents/GitHub/Cosmology/files/Powerspectrumm2phi2vec.txt', np.array([k, Ps(k,finGr,finGi), Pt(k,finhr,finhi),Pv(k,finvr,finvi), Pt(k,finhr,finhi)/Ps(k,finGr,finGi), ns]).T, delimiter='\t', fmt="%s",header='k    Ps    Pt    Pv    r    ns')
