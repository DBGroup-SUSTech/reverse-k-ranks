import numpy as np

# 只考虑可以merge user的情况
size_double = 8
size_int = 4
size_char = 1

n_user = 6218834
n_data_item = 2166750
topt = 1000000
n_interval = 128

description = 'merge by interval, store ID, bitmap'
size = n_interval * np.ceil(n_user / 8) / 1024 / 1024
print("{} {}MB".format(description, size))

description = 'merge by interval, store ID, bloom filter'
n = n_user
m = np.ceil(n / 2)
# k = np.ceil(np.log(2) * n / m)
k = 6
fp = (1 - np.exp(-k * n / m)) ** k

size = size_char * n_interval * np.ceil(m / 8) / 1024 / 1024
print("{} {}MB, false positive rate {}, k {}, m {}, n_user {}".format(description, size, fp, k, m, n_user))

description = 'merge by item, store score, all'
size = n_data_item * 2 * size_double / 1024 / 1024
print("{} {}MB".format(description, size))

description = 'merge by item, store ID, all'
size = n_data_item * 2 * size_char / 1024 / 1024
print("{} {}MB".format(description, size))

description = 'merge by item, store score, top-t'
size = (topt * 2 * size_double + np.ceil(n_data_item / 8)) / 1024 / 1024
print("{} {}MB".format(description, size))

description = 'merge by item, store ID, top-t'
size = (topt * 2 * size_char + np.ceil(n_data_item / 8)) / 1024 / 1024
print("{} {}MB".format(description, size))
