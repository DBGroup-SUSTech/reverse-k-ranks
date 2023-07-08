import numpy as np
import matplotlib.pyplot as plt
import os

dataset_m = {
    'movielens-27m': [283228, 53889],
    'netflix': [480189, 17770],
    'yahoomusic_big': [1948882, 98213],
    'yelp': [2189457, 160585],
    'book-crossing': [1, 2],
}


def show_bin_hist(bins, hist, name, dataset_name):
    # 直方图会进行统计各个区间的数值
    print(bins)
    fig, ax = plt.subplots()
    ax.bar(bins, hist, color='#b2b2b2', width=100)  # alpha设置透明度，0为完全透明

    # ax.set(xlim=(-5, 10), xticks=np.arange(-5, 10),   #)
    # ylim=(0, 1e8), yticks=np.arange(10000000, 90000000))
    n_user = dataset_m[dataset_name][0]
    n_data_item = dataset_m[dataset_name][1]
    # ax.set_yscale('log')
    # ax.set_title(
    #     '{}, user: {}, item: {}'.format(dataset_name, n_user, n_data_item))
    ax.set_xlabel('rank')
    ax.set_ylabel('frequency')
    # plt.xlim(0, 100)  # 设置x轴分布范围
    plt.savefig('{}.jpg'.format(name), dpi=600)
    plt.close()


if __name__ == '__main__':
    # dataset_name_l = ['movielens-27m', 'netflix', 'yahoomusic_big', 'yelp', 'book-crossing']
    dataset_name_l = ['movielens-27m']
    # for file in os.listdir('../../result/attribution'):
    for dataset_name in dataset_name_l:
        file_dir = '../../result/attribution/PrintUserRank/print-user-rank-{}.csv'.format(dataset_name)
        arr = np.loadtxt(file_dir, delimiter=',')
        n_bin = len(arr[0])
        arr_idx_l = np.arange(0, n_bin * 512, 512)
        # new_arr = arr[arr_idx_l]
        show_bin_hist(arr_idx_l, arr[0], dataset_name, dataset_name)
