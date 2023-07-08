import numpy as np
import matplotlib.pyplot as plt

# dataset_name = 'movielens-27m'

dataset_stat_m = {
    # n_user, n_data_item, n_query_item
    'fake-normal': [1000, 5000, 100],
    'netflix': [480189, 17770, 1000],
    'movielens-27m': [283228, 53889, 1000],
}


def show_bin_hist(bins, hist, dataset_name, topk):
    # 直方图会进行统计各个区间的数值
    print(bins)
    fig, ax = plt.subplots()
    ax.bar(bins, hist, color='#b2b2b2', width=30)  # alpha设置透明度，0为完全透明

    # ax.set(xlim=(-5, 10), xticks=np.arange(-5, 10),   #)
    # ylim=(0, 1e8), yticks=np.arange(10000000, 90000000))
    # n_user = dataset_m[dataset_name][0]
    # n_data_item = dataset_m[dataset_name][1]
    # ax.set_yscale('log')
    ax.set_title(
        'rank distribution, dataset: {}'.format(dataset_name))
    ax.set_xlabel('rank')
    ax.set_ylabel('frequency')
    # plt.xlim(0, 100)  # 设置x轴分布范围
    plt.savefig('rank-distribution-{}-top{}.jpg'.format(dataset_name, topk), dpi=600)
    np.savetxt('rank-distribution-{}-top{}.txt'.format(dataset_name, topk), hist, fmt="%d")
    plt.close()


if __name__ == '__main__':
    # for dataset_name in ['fake-normal', 'netflix', 'movielens-27m']:
    for dataset_name in ['netflix']:
        for topk in [10, 50]:
            n_user = dataset_stat_m[dataset_name][0]
            n_data_item = dataset_stat_m[dataset_name][1]
            n_query_item = dataset_stat_m[dataset_name][2]
            count_l = np.zeros(shape=n_data_item, dtype=np.int32)
            for qID in range(n_query_item):
                fname = '../../result/attribution/Candidate-{}/{}-top{}-qID-{}.txt'.format(dataset_name, dataset_name,
                                                                                           topk,
                                                                                           qID)
                print("qID {}".format(qID))
                with open(fname, 'r') as f:
                    lines = f.read().split('\n')
                    for line in lines:
                        if ':' in line:
                            arr = line.split(":")
                            userID = arr[0]
                            lb_rank = int(arr[1])
                            ub_rank = int(arr[2])
                            count_l[ub_rank:lb_rank] += 1
            bin_l = np.arange(n_data_item)
            show_bin_hist(bin_l, count_l, dataset_name, topk)
