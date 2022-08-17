import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

a = [0,0.4,0.5,0.2,0.1,1,0.22,0.9,0.3,0.12]
b = np.linspace(1,2,10)

#A, nu, k = 10, 4, 2

#def f(x, A, nu, k):
#    return A * np.exp(-k*x) * np.cos(2*np.pi * nu * x)

#xmax, nx = 0.5, 8
x = a #np.linspace(0, xmax, nx)
y = b #f(x, A, nu, k)

f_nearest = interp1d(x, y, kind='nearest')
f_linear  = interp1d(x, y)
f_cubic   = interp1d(x, y, kind='cubic')

x2 = np.linspace(0, 1, 100)

plt.plot(x, y, 'o', label='data points')
#plt.plot(x2, f(x2, A, nu, k), label='exact')
plt.plot(x2, f_nearest(x2), label='nearest')
plt.plot(x2, f_linear(x2), label='linear')
plt.plot(x2, f_cubic(x2), label='cubic')
plt.legend()
plt.show()
