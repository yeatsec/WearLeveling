import numpy as np
import matplotlib.pyplot as plt
import csv

task = 'go'
filename = 'cache_{}.txt'.format(task)
num_sets_l1 = 256
num_sets_l2 = 1024

gran_l1 = 2**8
thresh = 100000

cachefile = open(filename, 'r')
cachecsv = csv.reader(cachefile, delimiter=' ')

stat = {}

def get_key(lst, value):
    for key, val in enumerate(lst):
        if value == val:
            return key
    return None

def f(x):
    x = x // 4 # 4 bytes per word
    x = x // 8 # block size of 32 bytes in L1 data cache or L1 insn cache
    x = x % num_sets_l1 # 256 sets in either L1 cache with associativity of 1
    return x

def g(x, shift):
    x = f(x)
    mapbin = x//shift['gran']
    offset = x-(mapbin*shift['gran'])
    mapbin = shift['map'][mapbin] # physical location
    # get number of writes to this bin
    bin_writes = shift['wcounts'][mapbin]
    if (bin_writes > min(shift['wcounts']) + shift['thresh']):
        # swap the mappings
        shift['num_remap'] += 1
        tempbin = np.argmin(shift['wcounts']) # physical location
        minbin = get_key(shift['map'], tempbin) # indirect location
        shift['map'][minbin] = mapbin # physical location
        shift['map'][x//shift['gran']] = tempbin # x//shift[] was mapbin previously
        shift['wcounts'][tempbin] += gran_l1 # swapping the whole gran group
        shift['wcounts'][mapbin] += gran_l1 # swapping the whole gran group
        # account for additional writes to add to hist later
        # addl writes to mapbin: incurred from the swap
        for i in range(gran_l1):
            shift['addl_counts'][mapbin*gran_l1 + i] += 1
            if i != offset: # tempbin + offset will be counted by return
                shift['addl_counts'][tempbin*gran_l1 + i] += 1 
        return tempbin*shift['gran'] + offset
    else:
        shift['wcounts'][mapbin] += 1 # just writing to one line here
        return mapbin*shift['gran'] + offset

for row in cachecsv:
    if not (row[0] in stat):
        stat[row[0]] = {'r': [], 
        'w': [0 for _ in range(num_sets_l1)], 
        'total': 0, 
        'shift': {
            'gran': gran_l1,
            'num_entries': num_sets_l1//gran_l1,
            'wcounts': [0 for x in range(num_sets_l1//gran_l1)],
            'map': [b for b in range(num_sets_l1//gran_l1)],
            'thresh': thresh,
            'num_remap': 0,
            'addl_counts': [0 for _ in range(num_sets_l1)]
            }, 
        'num_access': 0}
    stat[row[0]]['num_access'] += 1
    stat[row[0]][row[1]][(g(int(row[2]), stat[row[0]]['shift']))] += 1
    stat[row[0]]['total'] += 1

# L1 Data Cache
r_locs = np.array(stat['dl1']['r'], dtype=int)
w_locs = np.array(stat['dl1']['w'], dtype=int)
total = stat['dl1']['total']
plt.figure()
w_locs = [w_locs[i] + stat['dl1']['shift']['addl_counts'][i] for i in range(num_sets_l1)]
w_vals, w_bins, patches = plt.hist(x=range(num_sets_l1), bins=range(num_sets_l1+1), weights=w_locs, color='b', label='Writes', alpha=0.7)
# r_median = int(np.median(r_vals))
w_median = int(np.median(w_vals))
plt.title('Write Distribution for L1 Data Cache\nTask: {} Access Total: {}\nThreshold: {} Granularity: {} # Remaps: {}'.format(task, total, stat['dl1']['shift']['thresh'], gran_l1, stat['dl1']['shift']['num_remap']))
plt.xlabel('Physical Set Location')
plt.ylabel('Frequency')
#plt.legend()
plt.tight_layout()
plt.savefig('figuresfr/dl1_{}_sn{}_gran{}.png'.format(task, thresh, gran_l1), dpi=100)
np.save('datafr/dl1_{}_sn{}_gran{}_wfreq.npy'.format(task,  thresh, gran_l1), w_vals)
np.save('datafr/dl1_{}_sn{}_gran{}_wbins.npy'.format(task, thresh, gran_l1), w_bins)
# np.save('datafr/dl1_{}_sn{}_gran{}_rfreq.npy'.format(task, thresh, gran_l1), r_vals)
# np.save('datafr/dl1_{}_sn{}_gran{}_rbins.npy'.format(task, thresh, gran_l1), r_bins)

plt.show()
cachefile.close()
