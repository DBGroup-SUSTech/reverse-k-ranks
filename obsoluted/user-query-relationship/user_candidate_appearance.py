import numpy as np
import matplotlib.pyplot as plt

# n_user, n_item, n_query
dataset_m = {
    'fake-normal': [1000, 5000, 100],
    'movielens-27m': [283228, 52889, 1000],
    'movielens-small': [610, 9524, 200],
    'netflix': [480189, 16770, 1000],
    'netflix-small': [2000, 5000, 100],
}


def show_bin_hist(bins, hist, name):
    # 直方图会进行统计各个区间的数值
    fig, ax = plt.subplots()
    ax.bar(bins, hist, color='#828487')  # alpha设置透明度，0为完全透明

    # ax.set(xlim=(-5, 10), xticks=np.arange(-5, 10),  # )
    #        ylim=(0, 1e8), yticks=np.arange(10000000, 90000000))
    # ax.set_yscale('log')
    # ax.set_title(
    #     '{}, user: {}, item: {}'.format(dataset_name, n_user, n_data_item))

    ax.set_xlabel('# user')
    ax.set_ylabel('query frequency')
    # plt.xlim(0, 100)  # 设置x轴分布范围
    plt.savefig('{}.jpg'.format(name), dpi=600, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # for dataset_name in ['fake-normal']:
    for dataset_name in ['movielens-27m']:
        topk = 100
        fname = '../../result/attribution/UserQueryRelationship/{}-reverse-k-rank-sorted-frequency.csv'.format(
            dataset_name)
        user_sort_freq_l = np.loadtxt(fname, delimiter='\n', dtype=np.int32).reshape(-1)
        show_bin_hist(np.arange(len(user_sort_freq_l)), user_sort_freq_l,
                      '{}-top{}-reverse-k-rank-sorted-frequency'.format(dataset_name, topk))
