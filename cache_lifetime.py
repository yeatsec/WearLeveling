import numpy as np
import matplotlib.pyplot as plt

shift_nums = [1, 10, 100, 1000, 10000, 100000, 1000000]

its = 500
task = 'gcc'

sgn = lambda x: 1.0 if x >= 0.0 else -1.0
lifetime = lambda c, c0: 1. - 0.15*np.log10(c/c0) if c > c0 else 1.
lt = lambda x: lifetime(x, 10**8) if lifetime(x, 10**8) >= 0 else 0.0
ltvect = np.vectorize(lt, otypes=[float])

x = np.logspace(start=1, stop=12, num=1001, endpoint=True)
y = ltvect(x)

def prod_reduce(arr):
    prod = 1.0
    for elem in arr:
        prod *= elem
    return prod

probs = []
for sn in shift_nums:
    freq =float(its)* np.load('./data/dl1_{}_sn{}_wfreq.npy'.format(task, sn), allow_pickle=True)
    bins = np.load('./data/dl1_{}_sn{}_wbins.npy'.format(task, sn), allow_pickle=True)
    # plt.figure()
    # plt.hist(x=range(256), bins=bins, weights=freq)
    # plt.show()
    probs.append(prod_reduce(ltvect(freq)))

plt.figure()
plt.title('Lifetime Prediction for {} Iterations of {}\nwith Incremental Remapping'.format(its, task))
plt.semilogx(shift_nums, probs, linewidth=2)
plt.xlabel('Shift Number')
plt.ylabel('Probability entire Cache is still functional')
plt.savefig('inc_map_lifetime_{}_{}its.png'.format(task, its), dpi=100)
plt.show()