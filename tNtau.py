#import libraries
import numpy as np
import matplotlib.pyplot as plt
import sympy as sy
from scipy import integrate
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

def dtLCDM(a,Or,Om,Ok,Ol,H0):
	dadt = H0*a*((Or/a**4)+(Om/a**3)+(Ok/a**2)+Ol)**0.50e0
	dtda=1.0e0/dadt
	return dtda

#Planck
Or=2.47e-5/(67.36/100.0)**2
Om=0.3153
Ok = 0.0e0
Ol=1-Or-Om-Ok
H0=67.36 #(km/s)/Mpc

def integrand(x):
	Or=2.47e-5/(67.36/100.0)**2
	Om=0.3153
	Ok = 0.0e0
	Ol=1-Or-Om-Ok
	H0=67.36
	int = dtLCDM(x,Or,Om,Ok,Ol,H0)
	return int

C = 3.085678e22/1000/3.1556926e16
x = sy.Symbol("x")
x = sy.Integral(integrand(x),(x,0,1))
Age = x.evalf()*C
print("Age of the Universe today is:", Age,'Gigayears')

Presentage = Age
#a vs Age

x = sy.Symbol("x")
x = sy.Integral(integrand(x),(x,0,1))
Age = x.evalf()*C

Ages = []
val = 100
a = np.linspace(0, 2, val)
for i in a:
	x = sy.Symbol("x")
	x = sy.Integral(integrand(x),(x,0,i))
	Age = x.evalf()*C
	Ages.append(Age)

up = []
down = []
for i in Ages:
	if i < Presentage:
		down.append(i)
	else:
		up.append(i)

Present = []
if (Presentage - down[-1]) < (Presentage - up[0]):
	Present.append(down[-1])
else:
	Present.append(up[0])

##conformal time
tau = []
for i in range(val-1):
	x = Ages[1:i+2]
	y = 1/a[1:i+2]
	tau.append(integrate.simpson(y, x))

plt.figure(figsize=(12,9))
plt.plot(Ages[1:],tau)
plt.ylabel("$\\tau$")
plt.xlabel("t (in Gigayears)")
plt.axvline(Present, 0, 1, label='Present',color='red')
plt.title('Change in Conformal time with t')
plt.savefig('/home/barnali/Documents/GitHub/Cosmology/plots/tNtau_tau_vs_t.pdf')
plt.show()

##efolds
#adot = []
#for i in range(val-1):
#	adot.append((a[i+1]-a[i])/(Ages[i+1]-Ages[i]))

#H = adot/a[1:]
H = H0*((Or/a[1:]**4)+(Om/a[1:]**3)+(Ok/a[1:]**2)+Ol)**0.50e0  #(km/s)/Mpc
H = H*1.0221e-3 #Gigayears^-1
N = []
for i in range(val-1):
	x = Ages[1:i+2]
	y = H[0:i+1]
	N.append(integrate.simpson(y, x))

plt.figure(figsize=(12,9))
plt.plot(Ages[1:],N)
plt.ylabel("N")
plt.xlabel("t (in Gigayears)")
plt.axvline(Present, 0, 1, label='Present',color='red')
plt.title('Change in efold with t')
plt.savefig('/home/barnali/Documents/GitHub/Cosmology/plots/tNtau_N_vs_t.pdf')
plt.show()

plt.figure(figsize=(12,9))
plt.plot(Ages[1:],tau,label = "Conformal time")
plt.plot(Ages[1:],N,label = "efold")
plt.ylabel("N,$\\tau$")
plt.xlabel("t (in Gigayears)")
plt.axvline(Present, 0, 1, label='Present',color='red')
plt.title('Change in efold and Conformal time with t')
plt.legend()
plt.savefig('/home/barnali/Documents/GitHub/Cosmology/plots/tNtau_tauN_vs_t.pdf')
plt.show()

plt.figure(figsize=(12,9))
plt.plot(N,tau)
plt.yscale('log')
plt.ylabel("$\\tau$")
plt.xlabel("N")
plt.title('Change in Conformal time with efold')
plt.savefig('/home/barnali/Documents/GitHub/Cosmology/plots/tNtau_tau_vs_N.pdf')
plt.show()
