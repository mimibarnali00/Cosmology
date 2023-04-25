#import necessary modules
import numpy as np
import matplotlib.pyplot as plt
import sympy as sy

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
