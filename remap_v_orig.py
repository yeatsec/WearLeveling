import numpy as np
import matplotlib.pyplot as plt
import csv

cntl_wfreq = np.load('./data/dl1_gcc_sninf_wfreq.npy', allow_pickle = True)
inc_wfreq = np.load('./data/dl1_gcc_sn10000_wfreq.npy', allow_pickle = True)
flex_wfreq = np.load('./datafr/dl1_gcc_sn100000_gran1_wfreq.npy', allow_pickle = True)
num_sets_l1 = 256
num_sets_l2 = 1024

plt.figure()
c_vals, c_bins, patches = plt.hist(x=range(256), bins=range(num_sets_l1+1), color='b', label='Default', alpha=0.5, weights = cntl_wfreq)

i_vals, i_bins, patches = plt.hist(x=range(256), bins=range(num_sets_l1+1), color='r', label='Incremental', alpha=0.5, weights = inc_wfreq)

plt.title('Write Distribution: Incremental vs. Default for L1 Data Cache\nTask:gcc, Shift Number:10000')
plt.xlabel('Physical Set Location')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('dl1_gcc_sn10000_comparison.png', dpi=100)

plt.figure()

c_vals, c_bins, patches = plt.hist(x=range(256), bins=range(num_sets_l1+1), color = 'b', label='Default', alpha=0.5, weights = cntl_wfreq)

f_vals, f_bins, patches = plt.hist(x=range(256), bins=range(num_sets_l1+1), color='g', label='Flexible', alpha=0.5, weights = flex_wfreq)

plt.title('Write Distribution: Flexible vs. Default for L1 Data Cache\nTask:gcc, Shift Number:100000, Granularity:1')
plt.xlabel('Physical Set Location')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('dl1_gcc_sn100000_gran1_comparison.png', dpi=100)

plt.show()
