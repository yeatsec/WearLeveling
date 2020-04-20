import numpy as np
import matplotlib.pyplot as plt

# Write count registers for each bin
# Adder to update write counts
# Logic to find minimum write counter
# Adder to add threshold to minimum write counter
# Comparator to compare current bin's write count to min bin's write count
# Memory to store a bin during switching
# Registers to implement a table to map bins to physical locations
# Logic to find desired bin and offset into bin given a load/store
# Miscellaneous logic

granularity = [1, 2, 4, 8, 16, 32, 64]
count_bit_len = 4
block_size = 32
total_storage = []

for gran in granularity:
    count_storage = count_bit_len * (256/gran)
    
    table_bit_len = np.log2(256/gran)
    table_storage = (table_bit_len * (256/gran) * 2)/8
    
    temp_storage = gran * block_size
    
    total_storage.append(100*(count_storage + table_storage + temp_storage)/(32*256))
    
plt.figure()
plt.plot(granularity, total_storage)
plt.title('Storage Overhead vs. Granularity')
plt.xlabel('Granularity')
plt.ylabel('Storage Overhead (% of cache size in bytes)')
plt.tight_layout()
plt.savefig('storage_vs_gran.png', dpi=100)
plt.show()
