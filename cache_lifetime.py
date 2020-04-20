import numpy as np
import matplotlib.pyplot as plt

shift_nums = [1, 10, 100, 1000, 10000, 100000, 1000000]


task = 'go'
colors = ['b', 'r', 'g', 'm', 'k']

remaps = [2767435, 276743, 27674, 2767, 276, 27, 2, 1, 0] # anagram
its = [6000, 8000, 10000, 12000, 14000]
if task == 'gcc':
    remaps = [38812101, 3881210, 388121, 38812, 3881, 388, 38, 3, 0] # gcc
    its = [500, 600, 700, 800, 900]
if task == 'go':
    remaps = [44582605, 4458260, 445826, 44582, 4458, 445, 44, 4, 0] # go
    its = [400, 500, 600, 700, 800]

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

plt.figure()
plt.title('Lifetime Prediction for Various Iterations of {}\nwith Incremental Remapping'.format(task))

for i, it in enumerate(its):
    probs = []
    for j, sn in enumerate(shift_nums):
        freq = np.load('./data/dl1_{}_sn{}_wfreq.npy'.format(task, sn), allow_pickle=True)
        bins = np.load('./data/dl1_{}_sn{}_wbins.npy'.format(task, sn), allow_pickle=True)
        temp = [0 for _ in range(256)]
        for it_i in range(it):
            remap_num = remaps[j] * it_i
            temp = [temp[loc] + freq[(remap_num + loc) % 256] for loc in range(256)]
        # plt.figure()
        # plt.hist(x=range(256), bins=bins, weights=temp)
        # plt.show()
        probs.append(prod_reduce(ltvect(temp)))
    plt.semilogx(shift_nums, probs, linewidth=2, label='{} Iterations'.format(it), color=colors[i], alpha=0.4)
plt.xlabel('Shift Number')
plt.ylabel('Probability entire Cache is still functional')
plt.legend()
plt.savefig('inc_map_lifetime_{}.png'.format(task), dpi=100)
# plt.figure()
# plt.title('Lifetime Prediction without Wear Leveling')
# its = 50*np.arange(41)
# for i, task in enumerate(['anagram', 'gcc', 'go']):
#     prob_vs_its = []
#     freq = np.load('./data/dl1_{}_sninf_wfreq.npy'.format(task), allow_pickle=True)
#     for it in its:
#         prob_vs_its.append(prod_reduce(ltvect(freq*it)))
#     plt.plot(its, prob_vs_its, linewidth=2, alpha=0.4, color=colors[i], label=task)
# plt.xlabel('Iterations')
# plt.ylabel('Probability entire Cache is still functional')
# plt.ylim((-0.05, 1.05))
# plt.legend()
# plt.savefig('inc_map_lifetime_control.png'.format(task, dpi=100))
plt.show()