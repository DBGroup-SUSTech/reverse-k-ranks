# 随机生成10000个点, 一个切分成两份, 一个切分成16份, 都是从[-2\sigma, 2\sigma]开始切分, 查看理论值和实际值的相差情况
import numpy as np
from scipy.stats import norm


def experimental_partition(arr, n_interval):
    max_itv = 2
    min_itv = -2
    itv_dist = (max_itv - min_itv) / n_interval
    larger_l = []
    for i in range(n_interval + 1):
        itv_val = max_itv - itv_dist * i
        n_large = sum(arr > itv_val)
        larger_l.append((itv_val, n_large))
    # for itv_val, n_large in larger_l:
    #     print("itv_val {}, n_large {}".format(itv_val, n_large))
    return larger_l


def theoretical_partition(n_arr, n_interval):
    max_itv = 2
    min_itv = -2
    itv_dist = (max_itv - min_itv) / n_interval
    larger_l = []
    for i in range(n_interval + 1):
        itv_val = max_itv - itv_dist * i
        n_large = n_arr * (1 - norm.cdf(itv_val))
        larger_l.append((itv_val, n_large))
    # for itv_val, n_large in larger_l:
    #     print("itv_val {}, n_large {}".format(itv_val, n_large))
    return larger_l


def eval_partition(exp_l, the_l):
    assert len(exp_l) == len(the_l)
    avg_num = np.average(np.absolute([exp_l[i][1] - the_l[i][1] for i in range(len(exp_l))]))
    print("average value difference {}, n_partition {}".format(avg_num, len(exp_l) - 1))


random_arr = np.random.normal(loc=0, scale=1.0, size=100000)
exp_l = experimental_partition(random_arr, 16)
the_l = theoretical_partition(len(random_arr), 16)
eval_partition(exp_l, the_l)
exp_l = experimental_partition(random_arr, 64)
the_l = theoretical_partition(len(random_arr), 64)
eval_partition(exp_l, the_l)
