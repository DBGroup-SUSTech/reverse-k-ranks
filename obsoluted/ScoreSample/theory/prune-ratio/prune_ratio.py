from scipy.stats import norm
import numpy as np

'''
first return k / n_user, second return i value, thrid return prune_ratio
'''


def est_prune_ratio(n_user, n_item, k, n_interval, bound_num):
    ratio = k / n_user

    if ratio >= 1 - norm.cdf(-bound_num):
        return ratio, np.inf, 0
    elif 1 - norm.cdf(bound_num) < ratio < 1 - norm.cdf(-bound_num):
        itvID = np.ceil(n_interval / 2 - n_interval / (bound_num * 2) * norm.ppf(1 - ratio))
        prune_ratio = norm.cdf(bound_num - bound_num * 2 / n_interval * itvID)
        return ratio, itvID, prune_ratio
    elif ratio <= 1 - norm.cdf(bound_num):
        prune_ratio = norm.cdf(bound_num)
        return ratio, 0, prune_ratio
    else:
        raise Exception("error")


n_user = 480189
n_item = 17770
k = 40
n_interval = 16
bound_num = 4

print("1 - norm.cdf(-{}) {}, 1 - norm.cdf({}) {}".format(bound_num, 1 - norm.cdf(-4), bound_num, 1 - norm.cdf(4)))

for k in [10, 20, 30, 40, 50, 60, 100, 500, 1000, 2000, 5000, 10000, 20000, 100000, 200000, 300000, 400000]:
    k_ratio, itvID, prune_ratio = est_prune_ratio(n_user, n_item, k, n_interval, bound_num)
    print("k {}, k / n_user {}, itvID {}, prune_ratio {}".format(k, k_ratio, itvID, prune_ratio))
