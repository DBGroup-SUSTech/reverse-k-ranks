import numpy as np
import matplotlib.pyplot as plt

# dataset_name = 'movielens-27m'

dataset_stat_m = {
    # n_user, n_data_item, n_query_item
    'fake-normal': [1000, 5000, 100],
    'netflix': [480189, 17770, 1000],
    'movielens-27m': [283228, 53889, 1000],
}


def show_bin_hist(bins, hist, dataset_name, topk, other_name):
    # 直方图会进行统计各个区间的数值
    print(bins)
    fig, ax = plt.subplots()
    ax.bar(bins, hist, color='fuchsia', width=1)  # alpha设置透明度，0为完全透明

    # ax.set(xlim=(-5, 10), xticks=np.arange(-5, 10),   #)
    # ylim=(0, 1e8), yticks=np.arange(10000000, 90000000))
    # n_user = dataset_m[dataset_name][0]
    # n_data_item = dataset_m[dataset_name][1]
    # ax.set_yscale('log')
    ax.set_title(
        'rank distribution, dataset: {}'.format(dataset_name))
    ax.set_xlabel('user distance')
    ax.set_ylabel('frequency')
    # plt.xlim(0, 100)  # 设置x轴分布范围
    plt.savefig('user-distance-distribution-{}-top{}{}.jpg'.format(dataset_name, topk, other_name))
    plt.close()


if __name__ == '__main__':
    topk = 10
    for dataset_name in ['fake-normal']:
        queryID_l = [0, 12, 24, 31, 37]
        for qID in queryID_l:
            distribution_l = np.loadtxt(
                '../../result/attribution/user-score-distribution-{}-top{}-queryID-{}.csv'.format(
                    dataset_name, topk, qID),
                delimiter=",")
            show_bin_hist(distribution_l[:, 0], distribution_l[:, 1], dataset_name, 10, "-queryID-{}".format(qID))
        distribution_l = np.loadtxt(
            '../../result/attribution/user-score-distribution-{}-top{}.csv'.format(dataset_name, topk), delimiter=",")
        show_bin_hist(distribution_l[:, 0], distribution_l[:, 1], dataset_name, 10, "")
