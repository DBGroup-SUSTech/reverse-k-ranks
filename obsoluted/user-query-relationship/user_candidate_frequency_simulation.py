import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib

params = {
    'axes.labelsize': 8,
    'font.size': 8,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': False,
    'figure.figsize': [4.5, 4.5]
}
matplotlib.RcParams.update(params)

# n_user, n_data_item, n_query
dataset_m = {
    'fake-normal': [1000, 5000, 100],
    'movielens-27m': [283228, 52889, 1000],
    'movielens-small': [610, 9524, 200],
    'netflix': [480189, 16770, 1000],
    'netflix-small': [2000, 5000, 100],
    'yahoomusic_big': [1823179, 135736, 1000],
    'yelp': [2189457, 160585, 1000],
}


def show_bin_hist(hist):
    # fig, ax = plt.subplots(figsize=(6, 6))
    bins = np.arange(0, len(hist), 1)  # 设置连续的边界值，即直方图的分布区间[0,10],[10,20]...
    # 直方图会进行统计各个区间的数值
    # plt.bar(bins, hist, len(bins), color='#828487')  # alpha设置透明度，0为完全透明
    plt.bar(bins, hist, width=1, color='#828487')  # alpha设置透明度，0为完全透明

    plt.xlabel('# user')
    plt.ylabel('query frequency')
    # plt.yscale('log')
    # plt.xlim(0, 100)  # 设置x轴分布范围
    plt.savefig(
        '{}-top{}-n_query_{}-reverse-k-rank-sorted-frequency-simulation.jpg'.format(dataset_name, topk, n_query),
        dpi=600, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    for dataset_name in ['movielens-27m', 'netflix', 'yahoomusic_big', 'yelp']:
        topk = 100
        n_user = dataset_m[dataset_name][0]
        n_query = dataset_m[dataset_name][2]
        freq_user = np.zeros(shape=n_user, dtype=np.int32)
        for i in range(n_query):
            randomlist = random.sample(range(0, n_user), topk)
            for userID in randomlist:
                freq_user[userID] += 1
        freq_user_sort = np.sort(freq_user)
        print("after sort")
        np.savetxt(
            '../../result/attribution/UserQueryRelationship/{}-top{}-reverse-k-rank-sorted-frequency-simulation.txt'.format(
                dataset_name, topk),
            freq_user_sort, fmt="%d")
        if dataset_name == 'movielens-27m':
            show_bin_hist(freq_user_sort)
