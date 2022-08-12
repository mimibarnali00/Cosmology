import numpy as np
import matplotlib.pyplot as plt

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

a = np.logspace(-10, 0, 1000)
Hz=np.zeros(len(a))
Hz = H(a,Or,Om,Ok,Ol,H0)*Km/MPc/C/mGeVinv/MplGeV
RadH = 1/Hz

lambda1=10**60*a
lambda2=10**57*a
lambda3=10**54*a

plt.xlabel(r'$\log_{10} a$')
plt.yscale('log')
plt.ylabel(r'$\frac{1}{H(a)} [1/M_{Pl}]$')
plt.plot(np.log10(a),RadH, lw=4)
plt.plot(np.log10(a),lambda1, lw=2,linestyle='--',label=r'$\lambda_1$')
plt.plot(np.log10(a),lambda2, lw=2,linestyle='--',label=r'$\lambda_2$')
plt.plot(np.log10(a),lambda3, lw=2,linestyle='--',label=r'$\lambda_3$')
plt.axvline(np.log10(1/1101), 0, 1, label='CMB',color='red')
plt.title('Physical lengthscales')
plt.legend()
plt.show()

plt.xlabel(r'$\log_{10} a$')
plt.yscale('log')
plt.ylabel(r'$\frac{1}{aH(a)} [1/M_{Pl}]$')
plt.plot(np.log10(a),RadH/a, lw=4)
plt.plot(np.log10(a),lambda1/a, lw=2,linestyle='--',label=r'$\lambda_1$')
plt.plot(np.log10(a),lambda2/a, lw=2,linestyle='--',label=r'$\lambda_2$')
plt.plot(np.log10(a),lambda3/a, lw=2,linestyle='--',label=r'$\lambda_3$')
plt.axvline(np.log10(1/1101), 0, 1, label='CMB',color='red')
plt.title('Comoving lengthscales')
plt.legend()
plt.show()
