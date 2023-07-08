import numpy as np
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


def show_bin_hist(hist, name):
    dataset_name = 'movielens-27m'
    n_user = 283228
    n_data_item = 53889
    print(hist)

    # fig, ax = plt.subplots(figsize=(6, 6))
    bins = np.arange(0, len(hist), 1)  # 设置连续的边界值，即直方图的分布区间[0,10],[10,20]...
    print(bins)
    bins *= 512
    print(bins)
    print(len(bins), len(hist))
    # 直方图会进行统计各个区间的数值
    plt.bar(bins, hist, len(bins) * 10, color='#828487')  # alpha设置透明度，0为完全透明

    plt.title(
        'rank distribution of a single query' + '\n{}, user: {}, item: {}'.format(dataset_name, n_user,
                                                                                               n_data_item))
    plt.xlabel('rank')
    plt.ylabel('number of user')
    # plt.yscale('log')
    # plt.xlim(0, 100)  # 设置x轴分布范围
    plt.savefig('RankSampleDisadvantage.pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    arr = np.loadtxt('../../../result/attribution/PrintUserRank/print-user-rank-movielens-27m.csv', delimiter=',')
    # avg_arr = [np.average(arr[:, i]) for i in range(len(arr[0]))]
    show_bin_hist(arr[9], 'fig-{}'.format(1))
