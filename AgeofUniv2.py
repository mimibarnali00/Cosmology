#import libraries
import numpy as np
import matplotlib.pyplot as plt
import sympy as sy
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
H0=67.36

a = np.linspace(0, 1, 1000)
plt.figure(figsize=(12,9))
plt.plot(a,dtLCDM(a,Or,Om,Ok,Ol,H0))
plt.xlabel("a(scale factor)")
plt.ylabel("dt/da(a)")
plt.title('Change in scale factor with time')
plt.show()

def integrand(x):
	int = dtLCDM(x,Or,Om,Ok,Ol,H0)
	return int

C = 3.085678e22/1000/3.1556926e16
x = sy.Symbol("x")
x = sy.Integral(integrand(x),(x,0,1))
Age = x.evalf()*C
print("Age of the Universe today is:", Age,'Gigayears')

#Age vs Curvature (k)
Ages = []
ok = np.linspace(-1,1,100)
for i in ok:
	def integrand(x):
		Ok = i
		Ol=1-Or-Om-Ok
		int = dtLCDM(x,Or,Om,Ok,Ol,H0)
		return int

	C = 3.085678e22/1000/3.1556926e16
	x = sy.Symbol("x")
	x = sy.Integral(integrand(x),(x,0,1))
	Age = x.evalf()*C
	Ages.append(Age)

plt.figure(figsize=(12,9))
plt.plot(ok,Ages)
plt.xlabel("Curvature ($\Omega_{k}$)")
plt.ylabel("Age (in Gigayears)")
plt.title('Change in Age with Curvature')
plt.savefig('/home/barnali/Documents/GitHub/Cosmology/plots/AgeofUniv2_age_vs_Omega_k.pdf')
plt.show()
