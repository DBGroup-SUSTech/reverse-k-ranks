import os.path
import struct
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({'font.size': 35})
hatch = ['--', '+', 'x', '\\']
width = 0.35  # the width of the bars: can also be len(x) sequence


def get_sample_ip_l(*, file_name: str, userid_l: list):
    sizeof_size_t = 8
    sizeof_int = 4
    sizeof_double = 8

    f = open(
        file_name,
        'rb')

    n_sample_b = f.read(sizeof_size_t)
    n_data_item_b = f.read(sizeof_size_t)
    n_user_b = f.read(sizeof_size_t)

    n_sample = struct.unpack("N", n_sample_b)[0]
    n_data_item = struct.unpack("N", n_data_item_b)[0]
    n_user = struct.unpack("N", n_user_b)[0]
    print(n_sample)
    print(n_data_item, n_user)

    # userid_l = np.random.choice(n_user, 20, replace=False)

    sample_arr_m = {}

    for userID in userid_l:
        f.seek(sizeof_size_t * 3 + sizeof_int * n_sample + userID * n_sample * sizeof_double, 0)
        sample_ip_l_b = f.read(n_sample * sizeof_double)
        sample_ip_l = struct.unpack("d" * n_sample, sample_ip_l_b)
        sample_arr_m[userID] = sample_ip_l

    f.close()
    return sample_arr_m


dataset_m = {'movielens-27m': [52889, 1000, 283228],
             'netflix': [16770, 1000, 480189],
             'yahoomusic_big': [135736, 1000, 1823179],
             'yahoomusic': [97213, 1000, 1948882],
             'yelp': [159585, 1000, 2189457],
             'goodreads': [2359650, 1000, 876145],
             'amazon-home-kitchen': [409243, 1000, 2511610],
             'yahoomusic_big_more_query': [135736, 1000, 1823179],
             'yelp_more_query': [159585, 1000, 2189457], }


def get_score_table_ip_l(*, file_name: str,
                         dataset_name: str,
                         userid_l: list):
    sizeof_double = 8

    f = open(file_name, 'rb')
    n_user = dataset_m[dataset_name][2]
    n_data_item = dataset_m[dataset_name][0]

    print(n_data_item, n_user)
    sample_arr_m = {}

    for userID in userid_l:
        f.seek(userID * n_data_item * sizeof_double, 0)
        sample_ip_l_b = f.read(n_data_item * sizeof_double)
        sample_ip_l = struct.unpack("d" * n_data_item, sample_ip_l_b)
        sample_arr_m[userID] = sample_ip_l

    f.close()
    return sample_arr_m


def plot_figure(*, method_name: str,
                score_l: list,
                set_yformatter: bool,
                is_test: bool):
    # fig = plt.figure(figsize=(25, 4))
    fig = plt.figure(figsize=(6, 4))

    subplot_str = 111
    ax = fig.add_subplot(subplot_str)

    ax.hist(score_l, color='#b2b2b2', bins=50)

    ax.set_xlabel('Score')
    ax.set_ylabel('Frequency')
    # ax.legend(frameon=False, bbox_to_anchor=(0.5, 1), loc="center", ncol=len(dataset_l), borderaxespad=5)
    # ax.set_xticks(np.arange(n_dataset), dataset_l)
    ax.set_xlim([-0.43, 2.3])
    # ax.set_ylim([0, 32])
    if set_yformatter:
        ax.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: 0 if x == 0 else str(int(x / 1000)) + 'k'))

    ax.margins(y=0.3)
    # fig.tight_layout(rect=(0.01, -0.07, 1.02, 1.05))
    if is_test:
        plt.savefig("query_aware_sample_distribution_{}.jpg".format(method_name), bbox_inches='tight', dpi=600)
    else:
        plt.savefig("query_aware_sample_distribution_{}.pdf".format(method_name), bbox_inches='tight')


def run_local(*, is_test: bool, userID: int):
    dir_name = '/home/bianzheng/reverse-k-ranks/index/memory_index_important'
    file_name = os.path.join(dir_name,
                             'QueryRankSampleSearchKthRank-yelp-n_sample_490-n_sample_query_5000-sample_topk_600.index')
    yelp_qrs_m = get_sample_ip_l(
        file_name=file_name,
        userid_l=[userID])

    file_name = os.path.join(dir_name,
                             'RankSample-yelp-n_sample_490.index')
    yelp_rs_m = get_sample_ip_l(
        file_name=file_name,
        userid_l=[userID])

    yelp_score_table_l = np.loadtxt(os.path.join(dir_name, f'yelp_score_table_userID_{userID}.txt'))

    score_l_l = [yelp_score_table_l, yelp_rs_m[userID], yelp_qrs_m[userID]]
    set_yformatter_l = [True, False, False]

    print(np.min(score_l_l[0]), np.min(score_l_l[1]), np.min(score_l_l[2]))
    print(np.max(score_l_l[0]), np.max(score_l_l[1]), np.max(score_l_l[2]))
    # score_l_l = [yahoomusic_qrs_m[3], yahoomusic_rs_m[3]]
    # title_l = ['Yahoomusic QAS', 'Yahoomusic US', 'Yelp QAS', 'Yelp US']
    method_l = ['score_table', 'uniform_sample', 'query_aware_sample']
    for score_l, method_name, set_yformatter in zip(score_l_l, method_l, set_yformatter_l):
        plot_figure(method_name=method_name, score_l=score_l, set_yformatter=set_yformatter, is_test=is_test)


def run_dbg_host(*, userID: int):
    dir_name = '/home/zhengbian/reverse-k-ranks/index'
    file_name = os.path.join(dir_name, 'yelp.index')
    yelp_score_table_m = get_score_table_ip_l(
        file_name=file_name,
        dataset_name='yelp', userid_l=[userID])

    score_l = yelp_score_table_m[userID]
    print(f"min score {np.min(score_l)}, max score {np.max(score_l)}")
    method_name = 'score_table'
    plot_figure(method_name=method_name, score_l=score_l, is_test=is_test)
    np.savetxt(f'yelp_score_table_userID_{userID}.txt', score_l)


if __name__ == '__main__':
    is_test = False

    userID = 8993
    # run_dbg_host(userID=userID)
    run_local(is_test=is_test, userID=userID)
