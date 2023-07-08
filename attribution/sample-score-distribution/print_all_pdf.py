import math
import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

'''n_item, n_query, n_user'''
dataset_m = {'fake-normal': [5000, 100, 1000],
             'fake-uniform': [5000, 100, 1000],
             'fakebig': [5000, 100, 5000],

             'movielens-1b': [272038, 1000, 2197225],
             'yahoomusic_big': [135736, 1000, 1823179],
             'yelp': [159585, 1000, 2189457],
             'amazon-electronics': [475002, 1000, 4201696],
             'amazon-home-kitchen': [409243, 1000, 2511610],
             'amazon-office-products': [305800, 1000, 3404914]}


def get_sample_ip_l(dataset):
    sizeof_size_t = 8
    sizeof_int = 4
    sizeof_double = 8

    f = open(
        '/home/bianzheng/reverse-k-ranks/index/movielens-27m.index', 'rb')

    n_data_item = dataset_m[dataset][0]
    n_user = dataset_m[dataset][2]

    print("{} n_data_item {}, n_user {}".format(dataset, n_data_item, n_user))

    # userid_l = np.random.choice(n_user, 20, replace=False)
    userid_l = np.arange(20)

    sample_arr_m = {}

    for userID in userid_l:
        f.seek(userID * sizeof_double * n_data_item, 0)
        sample_ip_l_b = f.read(sizeof_double * n_data_item)
        sample_ip_l = struct.unpack("d" * n_data_item, sample_ip_l_b)
        sample_arr_m[userID] = sample_ip_l

    f.close()
    return sample_arr_m


def show_hist(bins, dataset_name, name):
    # 直方图会进行统计各个区间的数值
    fig, ax = plt.subplots()
    ax.hist(bins, color='#b2b2b2', bins=100, width=0.1)
    # ax.bar(bins, hist, color='#b2b2b2', width=30)  # alpha设置透明度，0为完全透明

    # ax.set(xlim=(-5, 10), xticks=np.arange(-5, 10),   #)
    # ylim=(0, 1e8), yticks=np.arange(10000000, 90000000))
    # n_user = dataset_m[dataset_name][0]
    # n_data_item = dataset_m[dataset_name][1]
    # ax.set_yscale('log')
    ax.set_title(
        'All Sample IP Score Distribution, dataset: {}'.format(dataset_name))
    ax.set_xlabel('IP')
    ax.set_ylabel('frequency')
    # plt.xlim(0, 100)  # 设置x轴分布范围
    plt.savefig(name, dpi=600, bbox_inches='tight')
    plt.close()


sample_userID_ip_m = get_sample_ip_l('movielens-27m')
for i, userID in enumerate(sample_userID_ip_m.keys(), 0):
    ip_l = sample_userID_ip_m[userID]
    show_hist(ip_l, 'Movielens', 'pdf_all_sample_user_{}.jpg'.format(i))
