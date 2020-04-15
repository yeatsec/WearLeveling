import numpy as np
import matplotlib.pyplot as plt
import csv

task = 'go'
filename = 'cache_{}.txt'.format(task)
num_sets_l1 = 256
num_sets_l2 = 1024

shift_number = 1

cachefile = open(filename, 'r')
cachecsv = csv.reader(cachefile, delimiter=' ')

stat = {}

def f(x, shift):
    x = x // 4 # 4 bytes per word
    x = x // 8 # block size of 32 bytes in L1 data cache or L1 insn cache
    x = (x + shift) % num_sets_l1 # 256 sets in either L1 cache with associativity of 1
    return x

def g(x, shift):
    x = x // 4 # 4 bytes per word
    x = x // 16 # block size of 64 bytes in shared L2 data/insn cache
    x = (x + shift) % num_sets_l2
    return x

for row in cachecsv:
    if not (row[0] in stat):
        stat[row[0]] = {'r': [], 'w': [], 'total': 0, 'shift': 0, 'num_access': 0, 'num_remaps': 0}
    stat[row[0]]['num_access'] += 1
    if (row[0] == 'ul2'): # use g(x)
        stat[row[0]][row[1]].append(g(int(row[2]), stat[row[0]]['shift']))
    else: # use f(x)
        stat[row[0]][row[1]].append(f(int(row[2]), stat[row[0]]['shift']))
    stat[row[0]]['total'] += 1
    if stat[row[0]]['num_access'] >= shift_number:
    	stat[row[0]]['num_access'] = 0
    	stat[row[0]]['shift'] += 1 # don't need to wrap around for func sim
        stat[row[0]]['num_remaps'] += 1
        # for loc in range(num_sets_l1): gonna do this smarter
        #     stat[row[0]]['w'].append(loc)

# L1 Data Cache
r_locs = np.array(stat['dl1']['r'], dtype=int)
w_locs = np.array(stat['dl1']['w'], dtype=int)
num_remaps = stat['dl1']['num_remaps']
total = stat['dl1']['total']
w_vals, w_bins, patches = plt.hist(x=w_locs, bins=range(num_sets_l1+1), color='b', label='Writes', alpha=0.7)
r_vals, r_bins, patches = plt.hist(x=r_locs, bins=range(num_sets_l1+1), color='r', label='Reads', alpha=0.7)
w_vals = [val + num_remaps for val in w_vals] # account for the #remaps
plt.figure()
w_vals, w_bins, patches = plt.hist(x=range(num_sets_l1), bins=w_bins, weights=w_vals, color='b', label='Writes', alpha=0.7)
r_vals, r_bins, patches = plt.hist(x=range(num_sets_l1), bins=r_bins, weights=r_vals, color='r', label='Reads', alpha=0.7)
r_median = int(np.median(r_vals))
w_median = int(np.median(w_vals))
plt.title('Write Distribution for L1 Data Cache\nTask: {} Access Total: {} Shift After: {} Num Remaps: {}'.format(task, total, shift_number, num_remaps))
plt.xlabel('Physical Set Location')
plt.ylabel('Frequency')
#plt.legend()
plt.tight_layout()
plt.savefig('figures/dl1_{}_sn{}.png'.format(task, shift_number), dpi=300)
np.save('data/dl1_{}_sn{}_wfreq.npy'.format(task, shift_number), w_vals)
np.save('data/dl1_{}_sn{}_wbins.npy'.format(task, shift_number), w_bins)
# np.save('data/dl1_{}_sn{}_rfreq.npy'.format(task, shift_number), r_vals)
# np.save('data/dl1_{}_sn{}_rbins.npy'.format(task, shift_number), r_bins)

# # L1 Insn Cache
# r_locs = np.array(stat['il1']['r'], dtype=int)
# w_locs = np.array(stat['il1']['w'], dtype=int)
# total = stat['il1']['total']
# plt.figure()
# w_vals, w_bins, patches = plt.hist(x=w_locs, bins=range(num_sets_l1+1), color='r', label='Writes', alpha=0.7)
# r_vals, r_bins, patches = plt.hist(x=r_locs, bins=range(num_sets_l1+1), label='Reads', alpha=0.7)
# r_median = int(np.median(r_vals))
# w_median = int(np.median(w_vals))
# plt.title('Access Distribution for L1 Instruction Cache\nTask: {} Access Total: {} Shift After: {}\nMedian W Count: {} Median R Count: {}'.format(task, total, shift_number, w_median, r_median))
# plt.xlabel('Physical Set Location')
# plt.ylabel('Frequency')
# plt.legend()
# plt.tight_layout()
# plt.savefig('figures/il1_{}_sn{}.png'.format(task, shift_number), dpi=300)
# np.save('data/il1_{}_sn{}_wfreq.npy', w_vals)
# np.save('data/il1_{}_sn{}_wbins.npy', w_bins)
# np.save('data/il1_{}_sn{}_rfreq.npy', r_vals)
# np.save('data/il1_{}_sn{}_rbins.npy', r_bins)

# Shared L2 Data/Insn Cache
# r_locs = np.array(stat['ul2']['r'], dtype=int)
# w_locs = np.array(stat['ul2']['w'], dtype=int)
# total = stat['ul2']['total']
# plt.figure()
# w_vals, w_bins, patches = plt.hist(x=w_locs, bins=range(num_sets_l2+1), color='r', label='Writes', alpha=0.7)
# r_vals, r_bins, patches = plt.hist(x=r_locs, bins=range(num_sets_l2+1), label='Reads', alpha=0.7)
# r_median = int(np.median(r_vals))
# w_median = int(np.median(w_vals))
# plt.title('Access Distribution for L2 Data/Insn Cache\nTask: {} Access Total: {} Shift After: {}\nMedian W Count: {} Median R Count: {}'.format(task, total, shift_number, w_median, r_median))
# plt.xlabel('Physical Set Location')
# plt.ylabel('Frequency')
# plt.legend()
# plt.tight_layout()
# plt.savefig('figures/ul2_{}_sn{}.png'.format(task, shift_number), dpi=300)
# np.save('data/ul2_{}_sn{}_wfreq.npy', w_vals)
# np.save('data/ul2_{}_sn{}_wbins.npy', w_bins)
# np.save('data/ul2_{}_sn{}_rfreq.npy', r_vals)
# np.save('data/ul2_{}_sn{}_rbins.npy', r_bins)
plt.show()
cachefile.close()
