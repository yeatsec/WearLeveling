import numpy as np
import matplotlib.pyplot as plt



task = 'gcc'
thresh = 10000
gran = 2**2

freq = np.load('./datafr/dl1_{}_sn{}_gran{}_wfreq.npy'.format(task, thresh, gran), allow_pickle=True)
bins = np.load('./datafr/dl1_{}_sn{}_gran{}_wbins.npy'.format(task, thresh, gran), allow_pickle=True)

print(sum(freq))

plt.figure()
hfreq, bins, _ = plt.hist(x=range(256), bins=bins, weights=freq)
plt.show()