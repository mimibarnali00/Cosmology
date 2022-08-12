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
print("Age = ", Age)

Ages = []
a = np.linspace(0, 2, 100)
for i in a:
	x = sy.Symbol("x")
	x = sy.Integral(integrand(x),(x,0,i))
	Age = x.evalf()*C
	Ages.append(Age)
	print("scale factor = ", i, "Age = ", Age)

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

plt.plot(Ages,a)
plt.ylabel("a")
plt.xlabel("Age (in Gigayears)")
plt.axvline(Present, 0, 1, label='Present',color='red')
plt.title('Change in a with Age')
plt.show()
