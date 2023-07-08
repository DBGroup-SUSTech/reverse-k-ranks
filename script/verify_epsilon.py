import os
import struct
import matplotlib.pyplot as plt
import plot.query_aware_sample_distribution as query_aware_sample_distribution
import numpy as np
import matplotlib
import scipy.stats as stats

matplotlib.rcParams.update({'font.size': 35})
hatch = ['--', '+', 'x', '\\']
width = 0.35  # the width of the bars: can also be len(x) sequence


def get_sample_lr(*, file_name: str, userid_l: list):
    sizeof_size_t = 8
    sizeof_int = 4
    sizeof_double = 8

    f = open(file_name, 'rb')

    n_data_item_b = f.read(sizeof_size_t)
    n_user_b = f.read(sizeof_size_t)
    n_sample_rank_b = f.read(sizeof_int)

    n_data_item = struct.unpack("N", n_data_item_b)[0]
    n_user = struct.unpack("N", n_user_b)[0]
    n_sample_rank = struct.unpack("i", n_sample_rank_b)[0]

    print(n_sample_rank)
    print(n_data_item, n_user)

    # userid_l = np.random.choice(n_user, 20, replace=False)

    sample_arr_m = {}

    for userID in userid_l:
        f.seek(sizeof_size_t * 2 + sizeof_int + sizeof_int * n_sample_rank + userID * 2 * sizeof_double, 0)
        pred_para_l_b = f.read(2 * sizeof_double)
        pred_para_l = struct.unpack("d" * 2, pred_para_l_b)

        f.seek(sizeof_size_t * 2 + sizeof_int + sizeof_int * n_sample_rank + n_user * 2 * sizeof_double
               + userID * 2 * sizeof_double, 0)
        distribution_para_l_b = f.read(2 * sizeof_double)
        distribution_para_l = struct.unpack("d" * 2, distribution_para_l_b)

        f.seek(sizeof_size_t * 2 + sizeof_int + sizeof_int * n_sample_rank + n_user * 4 * sizeof_double
               + userID * sizeof_int, 0)
        error_b = f.read(sizeof_int)
        error = struct.unpack("i", error_b)

        sample_arr_m[userID] = [pred_para_l, distribution_para_l, error]

    f.close()
    return sample_arr_m


def plot_figure(*, method_name: str,
                score_l_l: list,
                rank_l_l: list,
                ylim_l: list,
                legend_loc: list,
                is_test: bool):
    # fig = plt.figure(figsize=(25, 4))
    fig = plt.figure(figsize=(6, 4))

    subplot_str = 111
    ax = fig.add_subplot(subplot_str)

    ax.plot(score_l_l[0], rank_l_l[0], color='#000000', linestyle='solid', linewidth=3, label='origin')
    ax.plot(score_l_l[1], rank_l_l[1], color='#000000', linestyle='dashed', linewidth=3, label='fit')

    # ax.legend(frameon=False, loc=legend_loc[0], bbox_to_anchor=legend_loc[1])

    ax.set_xlabel('Score')
    ax.set_ylabel('Rank')
    # ax.legend(frameon=False, bbox_to_anchor=(0.5, 1), loc="center", ncol=len(dataset_l), borderaxespad=5)
    # ax.set_xticks(np.arange(n_dataset), dataset_l)
    # ax.set_xlim([0, 2.5])
    ax.set_ylim(ylim_l)

    # ax.margins(y=0.3)
    # fig.tight_layout(rect=(0.01, -0.07, 1.02, 1.05))
    if is_test:
        plt.savefig("regression_based_pruning_{}.jpg".format(method_name), bbox_inches='tight', dpi=600)
    else:
        plt.savefig("regression_based_pruning_{}.pdf".format(method_name), bbox_inches='tight')


def run_local(*, userID: int, is_test: bool):
    dir_name = '/home/bianzheng/reverse-k-ranks/index/memory_index_important'

    file_name = os.path.join(dir_name,
                             'QueryRankSampleIntLR-yelp-n_sample_405-n_sample_query_5000-sample_topk_600.index')
    yelp_qrs_m = query_aware_sample_distribution.get_sample_ip_l(
        file_name=file_name, userid_l=[userID])

    score_l = yelp_qrs_m[userID]
    rank_l = np.arange(len(score_l))

    file_name = os.path.join(dir_name,
                             'MinMaxLinearRegression-yelp-n_sample_405.index')
    sample_parameter = get_sample_lr(file_name=file_name, userid_l=[userID])[userID]
    mu, sigma = sample_parameter[1]
    a, b = sample_parameter[0]

    predict_rank_l = [a * stats.norm.cdf((_ - mu) / sigma) + b for _ in score_l]
    print(f"error: {sample_parameter[2]}")


    method_name = 'before_transformation'
    ylim_l = [-10, 420]
    legend_loc = ['upper right', (1.05, 1.05)]
    plot_figure(method_name=method_name, score_l_l=[score_l, score_l], rank_l_l=[rank_l, predict_rank_l],
                ylim_l=ylim_l, legend_loc=legend_loc, is_test=is_test)


if __name__ == '__main__':
    userID = 8993
    is_test = True

    run_local(userID=userID, is_test=is_test)
