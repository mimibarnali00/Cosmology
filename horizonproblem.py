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

#CMB age
z = 1100
acmb = 1/1101
x = sy.Symbol("x")
x = sy.Integral(integrand(x),(x,0,acmb))
Age = x.evalf()*C

CMBage = Age
print("Age of the Universe at CMB is:", CMBage,'Gigayears')

#a vs Age
x = sy.Symbol("x")
x = sy.Integral(integrand(x),(x,0,1))
Age = x.evalf()*C

Ages = []
val = 10
a = np.linspace(0, 2, val)
for i in a:
	x = sy.Symbol("x")
	x = sy.Integral(integrand(x),(x,0,i))
	Age = x.evalf()*C
	Ages.append(Age)

#finding the closest age to Present age for plotting
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

#finding the closest age to CMB age for plotting
upc = []
downc = []
for i in Ages:
	if i < CMBage:
		downc.append(i)
	else:
		upc.append(i)

CMBa = []
if (CMBage - downc[-1]) < (CMBage - upc[0]):
	CMBa.append(downc[-1])
else:
	CMBa.append(upc[0])

##conformal time
tau = []
for i in range(val-1):
	x = Ages[1:i+2]
	y = 1/a[1:i+2]
	tau.append(integrate.simpson(y, x))

tauz = np.zeros(np.size(tau)+1)
tauz[1:] = tau
tau = tauz

plt.figure(figsize=(12,9))
plt.plot(Ages,tau)
plt.ylabel("Conformal time (in Gigayears)")
plt.xlabel("t (in Gigayears)")
plt.axvline(Present, 0, 1, label='Present',color='green')
plt.axvline(CMBa, 0, 1, label='CMB',color='red')
plt.title('Change in Conformal time with t')
plt.legend()
plt.savefig('/home/barnali/Documents/GitHub/Cosmology/plots/horizonproblem_tau_vs_t.pdf')
plt.show()

#plt.plot(a,Ages)
#plt.show()
#plt.loglog(tau,a[1:])
#plt.show()

Ages = np.array(Ages)
pindx = np.where(Ages == Present[0])[0][0]
tau0 = tau[pindx]
cindx = np.where(Ages == CMBa[0])[0][0]
taucmb = tau[cindx]
Xpos = []
for i in range(np.size(tau)):
	Xpos.append(-tau[i] + tau0) #y = mx + c; where m=-1, x=tau, c=tau0

Xneg = []
for i in range(np.size(tau)):
	Xneg.append(tau[i] - tau0)

print("tau_p - tau_cmb = ",tau0-taucmb)
print("tau_cmb - tau_begin = ",taucmb-tau[0])

plt.figure(figsize=(12,9))
plt.plot(Xpos,tau)
plt.plot(Xneg,tau)
#plt.grid()
plt.ylabel("$tau$ (in Gigayears)")
plt.xlabel("X")
plt.axhline(tau0, 0, 1, label='Present',color='green')
plt.axhline(taucmb, 0, 1, label='CMB',color='red')
#plt.axhline(y=0, color='black')
plt.axvline(x=0, color='black')
plt.legend()
plt.savefig('/home/barnali/Documents/GitHub/Cosmology/plots/horizonproblem.pdf')
plt.show()

print("tau_begining where tau_p - tau_cmb = tau_cmb - tau_begin is ",(2*taucmb) - tau0)

taunew_begin = (2*taucmb) - tau0
taunew = np.zeros(np.size(tau)+1)
taunew[1:] = tau
taunew[0] = taunew_begin

Xnewpos = np.zeros(np.size(Xpos)+1)
Xnewpos[1:] = Xpos
Xnewpos[0] = -taunew_begin + tau0

Xnewneg = np.zeros(np.size(Xneg)+1)
Xnewneg[1:] = Xneg
Xnewneg[0] = taunew_begin - tau0

Xnewposcmb = np.zeros(2)
Xnewposcmb[0] = 0
Xnewposcmb[1] = Xpos[0]

Xnewnegcmb = np.zeros(2)
Xnewnegcmb[0] = 0
Xnewnegcmb[1] = Xneg[0]

plt.figure(figsize=(12,9))
plt.plot(Xnewpos,taunew,color='blue')
plt.plot(Xnewneg,taunew,color='orange')
plt.plot(Xnewposcmb,taunew[0:2],color='blue')
plt.plot(Xnewnegcmb,taunew[0:2],color='orange')
plt.ylabel("$tau$ (in Gigayears)")
plt.xlabel("X")
plt.axhline(tau0, 0, 1, label='Present',color='green')
plt.axhline(taucmb, 0, 1, label='CMB',color='red')
plt.axvline(x=0, color='black')
plt.legend()
plt.savefig('/home/barnali/Documents/GitHub/Cosmology/plots/horizonproblem_inflation.pdf')
plt.show()

