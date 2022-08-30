import numpy as np
import matplotlib.pyplot as plt

a = np.linspace(1,2,10)
b = np.linspace(2,2.5,10)
c = np.linspace(3,4,10)

plt.figure()
plt.plot(a,b)

plt.figure()
plt.plot(b,c)
plt.show()
