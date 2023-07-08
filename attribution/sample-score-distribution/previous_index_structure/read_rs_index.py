import math
import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def get_sample_ip_l():
    sizeof_size_t = 8
    sizeof_int = 4
    sizeof_double = 8

    f = open(
        '/home/bianzheng/reverse-k-ranks/index/memory_index/RankSample-movielens-27m-n_sample_947.index',
        'rb')

    n_sample_b = f.read(sizeof_size_t)
    max_sample_every_b = f.read(sizeof_size_t)
    n_data_item_b = f.read(sizeof_size_t)
    n_user_b = f.read(sizeof_size_t)

    n_sample = struct.unpack("N", n_sample_b)[0]
    max_sample_every = struct.unpack("N", max_sample_every_b)[0]
    n_data_item = struct.unpack("N", n_data_item_b)[0]
    n_user = struct.unpack("N", n_user_b)[0]
    print(n_sample, max_sample_every)
    print(n_data_item, n_user)

    userid_l = np.random.choice(n_user, 20, replace=False)

    sample_arr_m = {}

    for userID in userid_l:
        f.seek(sizeof_size_t * 4 + sizeof_int * n_sample + userID * n_sample * sizeof_double, 0)
        sample_ip_l_b = f.read(n_sample * sizeof_double)
        sample_ip_l = struct.unpack("d" * n_sample, sample_ip_l_b)
        sample_arr_m[userID] = sample_ip_l

    f.close()
    return sample_arr_m


def plot(x_l, y_l, fname):
    plt.plot(x_l, y_l,
             color='#3D0DFF', linewidth=2.5, linestyle='-',
             label='Rank Sample',
             marker='H', markersize=2)
    plt.xlabel('sampled scores')
    # ax.set_ylabel('Running Time (ms)')
    # ax.set_ylim(0)
    # plt.yscale('log')
    plt.ylabel('rank')
    # plt.title('ip_plot')
    plt.savefig(fname, dpi=600)
    plt.close()


sample_userID_ip_m = get_sample_ip_l()
for i, userID in enumerate(sample_userID_ip_m.keys(), 0):
    ip_l = sample_userID_ip_m[userID]
    rank_l = np.arange(len(ip_l)) + 1
    plot(ip_l, rank_l, 'sample-ip-sample_user_{}.jpg'.format(i))
    mu = np.average(ip_l)
    std = np.std(ip_l)
    new_ip_l = [norm.cdf((x - mu) / std) for x in ip_l]
    new_rank_l = [(x - 1) / len(rank_l) for x in rank_l]
    plot(new_ip_l, rank_l, 'sample-ip-erf-sample_user_{}.jpg'.format(i))
