import numpy as np
import matplotlib.pyplot as plt

grans = [1, 2, 4, 8, 16, 32, 64, 128, 256]

its = 500
task = 'anagram'

sgn = lambda x: 1.0 if x >= 0.0 else -1.0
lifetime = lambda c, c0: 1. - 0.15*np.log10(c/c0) if c > c0 else 1.
lt = lambda x: lifetime(x, 10**7) if lifetime(x, 10**7) >= 0 else 0.0
ltvect = np.vectorize(lt, otypes=[float])

x = np.logspace(start=1, stop=12, num=1001, endpoint=True)
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

probs = []
for gran in grans:
    freq =float(its)* np.load('./datafr/dl1_{}_sn10000_gran{}_wfreq.npy'.format(task, gran), allow_pickle=True)
    bins = np.load('./datafr/dl1_{}_sn10000_gran{}_wbins.npy'.format(task, gran), allow_pickle=True)
    # plt.figure()
    # plt.hist(x=range(256), bins=bins, weights=freq)
    # plt.show()
    probs.append(prod_reduce(ltvect(freq)))

plt.figure()
plt.title('Lifetime Prediction for {} Iterations of {}\nwith Flexible Remapping'.format(its, task))
plt.semilogx(grans, probs, linewidth=2, basex=2)
plt.xlabel('Granularity')
plt.ylabel('Probability entire Cache is still functional')
plt.ylim((0, 1.05))
plt.savefig('flex_map_lifetime_{}_{}its.png'.format(task, its), dpi=100)
plt.show()