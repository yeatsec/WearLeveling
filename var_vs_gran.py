import numpy as np
import matplotlib.pyplot as plt
import csv

task = 'gcc'
granularity = [1, 2, 4, 8, 16, 32, 64, 128]
sn = '10000'
write_var = []
write_num = []
overhead = []
control = np.load('./data/dl1_{}_sninf_wfreq.npy'.format(task), allow_pickle = True)
control_tot = sum(control)
for gran in granularity:
    filename = './datafr/dl1_{}_sn{}_gran{}_wfreq.npy'.format(task, sn, str(gran))
    gran_wfreq = np.load(filename, allow_pickle = True)
    write_num.append(np.sum(gran_wfreq))
    write_var.append(np.sqrt(np.var(gran_wfreq)))
#print(write_num)
for idx in range(len(write_num)):
    overhead.append((write_num[idx] - control_tot) / control_tot)
#print(overhead)

plt.figure()
plt.plot(granularity, overhead)
plt.title('Overhead vs. Granularity for L1 Data Cache\nTask: {}, Shift Number: {}'.format(task, sn))
plt.xlabel('Granularity')
plt.ylabel('Overhead')
plt.tight_layout()
plt.savefig('dl1_{}_sn{}_overhead_v_gran.png'.format(task, sn), dpi=300)
plt.grid(True, which = 'both')

plt.figure()
plt.plot(granularity, write_var)
plt.title('Standard Deviation vs. Granularity for L1 Data Cache\nTask: {}, Shift Number: {}'.format(task, sn))
plt.xlabel('Granularity')
plt.ylabel('Std Dev Writes')
plt.gca().set_xscale('log', basex=2)
plt.tight_layout()
plt.savefig('dl1_{}_sn{}_var_v_gran.png'.format(task, sn), dpi=300)
plt.grid(True, which = 'both')

plt.show()

