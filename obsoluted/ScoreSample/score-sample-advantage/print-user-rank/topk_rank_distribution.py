import numpy as np
import matplotlib.pyplot as plt

# movielens
n_user = 283228
n_data_item = 53889


def show_freqency_hist(data):
    # plt.xlim((-3, 3))
    # plt.ylim((0, 0.5))
    '''
    plt.hist():
    参数bins: 划分间隔，可以采用 整数来指定间隔的数量，然后由程序由间隔的数量来确定每一个间隔的范围，也可以通过列表来直接指定间隔的范围
    density的类型是 bool型，若为True,则为频率直方图，反之，频数直方图:
        频率直方图，的在统计出每一个间隔中的频数后，将频数除以总的观测数，
        就得到了每一个间隔中的频率，然后将频率除以组距（每一个间隔的宽度），
        即用纵轴来表示 频率/组距 的大小；之所以要除以组距的目的是为了让频率直方图的阶梯形折线将逼近于概率密度曲线。
        也就是说，当观测数据充分大时，频率直方图近似地反映了概率密度曲线的大致形状，在统计推断中常常由此提出对总体分布形式的假设。
        频数直方图，就是给定桶的范围，查看有多少个落到这个桶的范围中
    stacked:如果为``True''，则多个数据相互堆叠
    '''
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(data, bins=50, density=False, stacked=False, facecolor='#bebebe', alpha=0.75,
                               edgecolor='#000000')

    ax.set_xlabel('rank')
    ax.set_ylabel('frequency')
    # ax.set_title('movielens-27m, n_user: {}, n_data_item {}'.format(n_user, n_data_item))

    plt.savefig('hist_frequency.jpg', dpi=600, bbox_inches = 'tight')
    plt.close()


if __name__ == '__main__':
    topk = 70
    dataset_name = 'movielens-27m'

    topk_arr = np.loadtxt(
        '/home/bianzheng/reverse-k-ranks/result/attribution/PrintUserRank/rank-top-{}-{}.csv'.format(topk,
                                                                                                     dataset_name),
        delimiter=',', dtype=np.int32)
    topk_arr = topk_arr[:, topk - 1]
    print(topk_arr)
    print(topk_arr.shape)

    show_freqency_hist(topk_arr)
