import numpy as np
import matplotlib.pyplot as plt
import csv

task = 'gcc'
shiftNums = [1, 10, 100, 1000, 10000, 100000, 1000000, np.inf]
write_var = []
write_num = []
overhead = []
for sn in shiftNums:
    filename = './data/dl1_{}_sn{}_wfreq.npy'.format(task, str(sn))
    sn_wfreq = np.load(filename, allow_pickle = True)
    write_num.append(np.sum(sn_wfreq))
    write_var.append(np.sqrt(np.var(sn_wfreq)))
for idx in range(len(write_num)):
    overhead.append((write_num[idx] - write_num[-1]) / write_num[-1])
#print(overhead)

plt.figure()
plt.semilogx(shiftNums, write_var)
plt.title('Standard Deviation vs. Shift Number for L1 Data Cache\nTask: {}'.format(task))
plt.xlabel('Shift Number')
plt.ylabel('Std Dev')
plt.tight_layout()
plt.savefig('dl1_{}_var_v_sn.png'.format(task), dpi=300)
plt.grid(True, which = 'both')

plt.figure()
plt.semilogx(shiftNums, overhead)
plt.title('Overhead vs. Shift Number for L1 Data Cache\nTask: {}'.format(task))
plt.xlabel('Shift Number')
plt.ylabel('Overhead')
plt.gca().set_yscale('log')
plt.tight_layout()
plt.savefig('dl1_{}_overhead_v_sn.png'.format(task), dpi=300)
plt.grid(True, which = 'both')

plt.show()

