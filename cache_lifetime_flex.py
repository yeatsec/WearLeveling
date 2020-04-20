import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

grans = [1, 2, 4, 8, 16, 32, 64, 128, 256]

its = [1000, 2000, 3000, 4000, 5000]
task = 'anagram'
sn = 10000

colors = ['b', 'r', 'g', 'm', 'k']

sgn = lambda x: 1.0 if x >= 0.0 else -1.0
lifetime = lambda c, c0: 1. - 0.15*np.log10(c/c0) if c > c0 else 1.
lt = lambda x: lifetime(x, 10**8) if lifetime(x, 10**8) >= 0 else 0.0
ltvect = np.vectorize(lt, otypes=[float])

x = np.logspace(start=1, stop=15, num=1001, endpoint=True)
y = ltvect(x)

plt.figure()
plt.semilogx(x, y, linewidth=2)
plt.title("Cache Line Functionality Probability vs Write Count")
plt.xlabel('Write Count')
plt.ylabel('Probability')
plt.savefig('func_dist.png', dpi=100)

def prod_reduce(arr):
    prod = 1.0
    for elem in arr:
        prod *= elem
    return prod

plt.figure()
plt.title('Lifetime Prediction for Various Iterations of {}\nwith Flexible Remapping'.format(task))

for i, it in enumerate(its):
    probs = []
    for gran in grans:
        freq = np.load('./datafr/dl1_{}_sn{}_gran{}_wfreq.npy'.format(task, sn, gran), allow_pickle=True)
        #ave_freq = mean(freq)
        #print(10**8/ave_freq)
        freq = [int(f_val*it) for f_val in freq]
        bins = np.load('./datafr/dl1_{}_sn{}_gran{}_wbins.npy'.format(task, sn, gran), allow_pickle=True)
        # plt.figure()
        # plt.hist(x=range(256), bins=bins, weights=freq)
        # plt.show()
        probs.append(prod_reduce(ltvect(freq)))
    plt.semilogx(grans, probs, linewidth=2, basex=2, alpha=0.4, color=colors[i], label='{} Iterations'.format(it))

plt.xlabel('Granularity')
plt.ylabel('Probability entire Cache is still functional')
plt.ylim((-0.05, 1.05))
plt.legend()
plt.savefig('flex_map_lifetime_{}.png'.format(task), dpi=100)
plt.show()