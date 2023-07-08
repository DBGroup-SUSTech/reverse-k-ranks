import struct
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.stats as stats

linestyle_l = ['_', '-', '--', ':']
color_l = ['#3D0DFF', '#6BFF00', '#00E8E2', '#EB0225', '#FF9E03']
marker_l = ['x', "v", "o", "D", "s"]
markersize = 5

matplotlib.rcParams.update({'font.size': 25})


def get_sample_ip_l(file_name, userid_l):
    sizeof_size_t = 8
    sizeof_int = 4
    sizeof_double = 8

    f = open(
        '/home/bianzheng/reverse-k-ranks/index/memory_index_important/{}.index'.format(file_name),
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


def get_sample_lr(file_name, userid_l):
    sizeof_size_t = 8
    sizeof_int = 4
    sizeof_double = 8

    f = open(
        '/home/bianzheng/reverse-k-ranks/index/memory_index_important/{}.index'.format(file_name),
        'rb')

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

    ax.plot(score_l_l[0], rank_l_l[0], color='#000000', linestyle='solid', linewidth=3, label='Score')
    # ax.scatter(x=score_l_l[0], y=rank_l_l[0], s=2, label='sampled score')
    ax.plot(score_l_l[1], rank_l_l[1], color='#000000', linestyle='dashed', linewidth=3, label='Fitting')

    ax.legend(frameon=False, loc=legend_loc[0], bbox_to_anchor=legend_loc[1])

    # for score_l, rank_l, i in zip(score_l_l, rank_l_l, np.arange(len(score_l_l))):
    #     ax.plot(score_l, rank_l, color='#b2b2b2', marker=marker_l[i], fillstyle='none', markersize=markersize)

    ax.set_xlabel('Score')
    ax.set_ylabel('Rank')
    # ax.legend(frameon=False, bbox_to_anchor=(0.5, 1), loc="center", ncol=len(dataset_l), borderaxespad=5)
    # ax.set_xticks(np.arange(n_dataset), dataset_l)
    # ax.set_xlim([0, 2.5])
    ax.set_ylim(ylim_l)

    # ax.margins(y=0.3)
    # fig.tight_layout(rect=(0.01, -0.07, 1.02, 1.05))
    if is_test:
        plt.savefig("ScoreRankCurve_{}.jpg".format(method_name), bbox_inches='tight', dpi=600)
    else:
        plt.savefig("ScoreRankCurve_{}.pdf".format(method_name), bbox_inches='tight')


amazon_id = 1000

amazon_pre_m = get_sample_lr('MinMaxLinearRegression-amazon-home-kitchen-n_sample_336-n_sample_query_5000-sample_topk_600',
                           [amazon_id])
amazon_qrs_m = get_sample_ip_l(
    'QueryRankSampleIntLR-amazon-home-kitchen-n_sample_336-n_sample_query_5000-sample_topk_600',
    [amazon_id])

score_l_l = [amazon_qrs_m[amazon_id]]
method_l = ['1_Amazon']
is_test = True

score_l = score_l_l[0]
method_name = method_l[0]
rank_l = np.arange(len(score_l))
amazon_pre = amazon_pre_m[amazon_id]
# ylim_l = [-20, 420]
ylim_l = None
print("second error {}".format(amazon_pre[2]))
rank_pred_l = [
    amazon_pre[0][0] * stats.norm.cdf((_ - amazon_pre[1][0]) / amazon_pre[1][1]) + amazon_pre[0][1]
    for _ in score_l
]
# rank_pred_error_l = [
#     amazon_pre[0][0] * stats.norm.cdf((_ - amazon_pre[1][0]) / amazon_pre[1][1]) + amazon_pre[0][1] + amazon_pre[2][0]
#     for _ in score_l
# ]
legend_loc = ['upper right', (1.05, 1.05)]
plot_figure(method_name=method_name, score_l_l=[score_l, score_l], rank_l_l=[rank_l, rank_pred_l],
            ylim_l=ylim_l, legend_loc=legend_loc, is_test=is_test)
