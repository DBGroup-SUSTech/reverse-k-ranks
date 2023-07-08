import struct
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.stats import norm

matplotlib.rcParams.update({'font.size': 20})
hatch = ['--', '+', 'x', '\\']
width = 0.35  # the width of the bars: can also be len(x) sequence

sizeof_size_t = 8
sizeof_int = 4
sizeof_double = 8
sizeof_float = 4


def get_sample_ip_l(file_name, userid_l):
    f = open(
        '/home/bianzheng/reverse-k-ranks/index/memory_index/amazon/{}'.format(file_name), 'rb')

    n_sample_b = f.read(sizeof_size_t)
    n_data_item_b = f.read(sizeof_size_t)
    n_user_b = f.read(sizeof_size_t)

    n_sample = struct.unpack("N", n_sample_b)[0]
    n_data_item = struct.unpack("N", n_data_item_b)[0]
    n_user = struct.unpack("N", n_user_b)[0]
    print(f"n_sample {n_sample}, n_data_item {n_data_item}, n_user {n_user}")

    # userid_l = np.random.choice(n_user, 20, replace=False)

    sample_arr_m = {}

    for userID in userid_l:
        f.seek(sizeof_size_t * 3 + sizeof_int * n_sample + userID * n_sample * sizeof_float, 0)
        sample_ip_l_b = f.read(n_sample * sizeof_float)
        sample_ip_l = struct.unpack("f" * n_sample, sample_ip_l_b)
        sample_arr_m[userID] = sample_ip_l

    f.close()
    return sample_arr_m


def show_hist(bins, dataset_name, name):
    # 直方图会进行统计各个区间的数值
    fig, ax = plt.subplots()
    ax.hist(bins, color='#b2b2b2', bins=50, width=0.1)
    # ax.bar(bins, hist, color='#b2b2b2', width=30)  # alpha设置透明度，0为完全透明

    # ax.set(xlim=(-5, 10), xticks=np.arange(-5, 10),   #)
    # ylim=(0, 1e8), yticks=np.arange(10000000, 90000000))
    # n_user = dataset_m[dataset_name][0]
    # n_data_item = dataset_m[dataset_name][1]
    # ax.set_yscale('log')
    ax.set_title('{}'.format(dataset_name))
    ax.set_xlabel('Score')
    ax.set_ylabel('Frequency')
    # plt.xlim(0, 100)  # 设置x轴分布范围
    plt.savefig('{}.jpg'.format(name), dpi=600, bbox_inches='tight')
    # plt.savefig('{}.pdf'.format(name), bbox_inches='tight')
    plt.close()


def plot_figure(*, result_fname: str,
                sample_score_l: np.ndarray,
                rank_l: np.ndarray, lim=None):
    assert len(sample_score_l) == len(rank_l)
    fig = plt.figure(figsize=(1 * 4 + 2, 4))

    subplot_str = int('111')
    ax = fig.add_subplot(subplot_str)
    ax.plot(sample_score_l, rank_l, color='#b2b2b2')

    ax.set_xlabel('Score')
    ax.set_ylabel('Rank')
    ax.set_xlim(lim)
    # ax.legend(frameon=False, bbox_to_anchor=(0.5, 1), loc="center", ncol=len(dataset_l), borderaxespad=5)
    # ax.set_xticks(np.arange(n_dataset), dataset_l)

    fig.tight_layout()
    plt.savefig("{}.jpg".format(result_fname), dpi=600)


def show_hist(bins, name):
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
        'All Sample IP Score Distribution')
    ax.set_xlabel('IP')
    ax.set_ylabel('frequency')
    # plt.xlim(0, 100)  # 设置x轴分布范围
    plt.savefig(name, dpi=600, bbox_inches='tight')
    plt.close()


def plot_sample(index_name: str, result_fname: str):
    sample_score_l = get_sample_ip_l(
        index_name,
        [10])[10]
    rank_l = np.arange(len(sample_score_l))
    plot_figure(sample_score_l=sample_score_l, rank_l=rank_l, result_fname=result_fname)

    new_result_fname = '{}-transform'.format(result_fname)

    mu = np.average(sample_score_l)
    std_val = np.std(sample_score_l)
    new_sample_score_l = np.array([norm.cdf((_ - mu) / std_val) for _ in sample_score_l])
    plot_figure(sample_score_l=new_sample_score_l, rank_l=rank_l, result_fname=new_result_fname)

    show_hist(sample_score_l, '{}-hist'.format(result_fname))


if __name__ == "__main__":
    index_name = 'QS-amazon-home-kitchen-n_sample_6840-n_sample_query_5000-sample_topk_200.index'
    result_fname = 'SampleDistribution-amazon-home-kitchen'
    plot_sample(index_name, result_fname)

    index_name = 'QS-movielens-1b-n_sample_7818-n_sample_query_5000-sample_topk_200.index'
    result_fname = 'SampleDistribution-movielens-1b'
    plot_sample(index_name, result_fname)

    index_name = 'QS-yelp-n_sample_7846-n_sample_query_5000-sample_topk_200.index'
    result_fname = 'SampleDistribution-yelp'
    plot_sample(index_name, result_fname)

    # result_fname = 'SampleDistribution_before'
    # rank_l = np.arange(len(sample_score_l))
    # plot_figure(sample_score_l=sample_score_l, rank_l=rank_l, result_fname=result_fname)
    #
    # x_l = np.linspace(start=-4, stop=4, num=100)
    # y_l = [-norm.cdf(_) for _ in x_l]
    # result_fname = 'simulation'
    # plot_figure(sample_score_l=x_l, rank_l=y_l, result_fname=result_fname)
