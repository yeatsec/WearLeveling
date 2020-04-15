import numpy as np
import matplotlib.pyplot as plt

shift_nums = [1, 10, 100, 1000, 10000, 100000, 1000000]
tau = 10**(-9.25)

lifetime = lambda w: (1./(1.+np.exp(tau*(w-10**10))))
ltvect = np.vectorize(lifetime, otypes=[float])

x = np.logspace(start=1, stop=12, num=101, endpoint=True)
y = ltvect(x)

fig = plt.figure()
plt.plot(x, y, linewidth=2)
ax = plt.gca()
ax.set_xscale('log')
plt.show()