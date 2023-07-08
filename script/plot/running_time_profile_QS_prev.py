import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
import matplotlib
import math

linestyle_l = ['_', '-', '--', ':']
color_l = ['#3D0DFF', '#6BFF00', '#00E8E2', '#EB0225', '#FF9E03']
marker_l = ['x', "v", "o", "D", "s"]
markersize = 15

hatch = ['--', '+', 'x', '\\']
width = 0.35  # the width of the bars: can also be len(x) sequence


def plot_figure_component(*, dataset_l: list,
                          fig_y: str, result_fname: str,
                          refine_time_l: list, score_compute_time_l: list,
                          rank_bound_compute_time_l: list,
                          is_test: bool):
    matplotlib.rcParams.update({'font.size': 25})
    n_dataset = len(dataset_l)
    # fig = plt.figure(figsize=(25, 4))
    fig = plt.figure(figsize=(10, 4))
    fig.text(0.01, 0.5, fig_y, va='center', rotation='vertical')

    subplot_str = int('111')
    ax = fig.add_subplot(subplot_str)

    refine_time_l = np.array(refine_time_l)
    score_compute_time_l = np.array(score_compute_time_l)
    rank_bound_compute_time_l = np.array(rank_bound_compute_time_l)

    total_time_l = refine_time_l + score_compute_time_l + rank_bound_compute_time_l

    refine_time_perc_l = refine_time_l / total_time_l
    score_compute_time_perc_l = score_compute_time_l / total_time_l
    rank_bound_compute_time_perc_l = rank_bound_compute_time_l / total_time_l
    assert len(refine_time_l) == len(score_compute_time_perc_l) == len(rank_bound_compute_time_perc_l)

    width = 0.35
    position_l = np.sort(np.append(np.arange(n_dataset) - width / 28 * 18, np.arange(n_dataset) + width / 28 * 18))
    assert len(refine_time_l) == len(position_l)

    ip_time_ins = ax.bar(position_l, score_compute_time_perc_l, width - 0.1, color='#000000', edgecolor='#000000',
                         hatch='', label='Compute Score')
    ip_time_ins = ax.bar(position_l, rank_bound_compute_time_perc_l, width - 0.1,
                         bottom=score_compute_time_perc_l,
                         color='#ffffff', edgecolor='#000000',
                         hatch='\\', label='Compute Rank Bound')
    io_time_ins = ax.bar(position_l, refine_time_perc_l, width - 0.1,
                         bottom=score_compute_time_perc_l + rank_bound_compute_time_perc_l,
                         color='#ffffff',
                         edgecolor='#000000',
                         hatch='', label='Refine')
    ax.set_ylim([0, 1.55])
    # ax.set_xlabel(dataset_l[fig_i])
    # ax.set_ylabel('')
    # ax.set_title(dataset_l[fig_i])
    # ax.legend(frameon=False, bbox_to_anchor=(0.5, 1), loc="center", ncol=len(dataset_l), borderaxespad=5)
    ax.legend(frameon=False, loc="upper center", ncol=len(dataset_l), borderaxespad=-0.2,
              # columnspacing=0.8, handletextpad=0.1, handlelength=1.5, labelspacing=0.1)
              labelspacing=0.2)
    # ax.set_xticks(np.arange(n_dataset), dataset_l)
    ax.set_xticks(np.arange(n_dataset))
    ax.set_xticklabels(dataset_l)
    ax.set_yticks([0, 0.5, 1.0])

    ax.bar_label(io_time_ins, labels=[r'k=10', 'k=200', 'k=10', 'k=200'], padding=0)

    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    # ax.margins(y=50)

    fig.tight_layout(rect=(0.01, -0.09, 1.03, 1.09))
    # fig.tight_layout()
    if is_test:
        plt.savefig("{}.jpg".format(result_fname), dpi=600)
    else:
        plt.savefig("{}.pdf".format(result_fname))


if __name__ == "__main__":
    dataset_l = ['Amazon', 'Yelp']
    fig_y = 'Pct.Query Time'
    is_test = False

    score_compute_time_l = [22.258, 21.018, 17.997, 18.222]
    rank_bound_compute_time_l = [45.395 + 8.603, 49.089 + 8.692, 57.121 + 7.523, 57.686 + 7.663]
    refine_time_l = [7.252, 33.607, 0.204, 2.945]
    result_fname = 'running_time_profile_QS'
    plot_figure_component(dataset_l=dataset_l,
                          fig_y=fig_y, result_fname=result_fname,
                          refine_time_l=refine_time_l, score_compute_time_l=score_compute_time_l,
                          rank_bound_compute_time_l=rank_bound_compute_time_l,
                          is_test=is_test)
