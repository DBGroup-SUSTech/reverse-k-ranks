import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import ticker
import numpy as np
import math

linestyle_l = ['_', '-', '--', ':']
color_l = ['#3D0DFF', '#6BFF00', '#00E8E2', '#EB0225', '#FF9E03']
marker_l = ['x', "v", "o", "D", "s"]
markersize = 15

matplotlib.rcParams.update({'font.size': 25})


def plot_figure(*, n_refine_user_l: list,
                xlim: list, ylim: list, x_ticks: list, n_bin: int,
                fname_sufix: str,
                name_m: dict, is_test: bool):
    # fig = plt.figure(figsize=(25, 4))
    fig = plt.figure(figsize=(6, 4))
    subplot_str = 111
    ax = fig.add_subplot(subplot_str)
    # counts, bins = np.histogram(total_time_l, bins=n_bin, weights=np.ones(len(total_time_l)) * weight)
    # print(counts)
    # assert 1000 - 0.1 <= np.sum(counts) <= 1000 + 0.1
    # ax.stairs(counts, bins, color='#828487', fill=True)
    # print(np.logspace(2, 5, num=100))
    ax.hist(n_refine_user_l, bins=n_bin,
            color='#828487')
    ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='x')

    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(name_m['fig_x'])
    ax.set_ylabel(name_m['fig_y'])
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if x_ticks:
        ax.set_xticks(x_ticks)
    ax.set_yticks([1e0, 1e1, 1e2, 1e3])

    ax.set_xticks([0, 25, 50, 75])

    from matplotlib.ticker import ScalarFormatter
    xfmt = ScalarFormatter()
    xfmt.set_powerlimits((-3, 3))
    ax.get_xaxis().set_major_formatter(xfmt)

    if is_test:
        plt.savefig(f"rmips_result_size_{fname_sufix}.jpg", bbox_inches='tight')
    else:
        plt.savefig(f"rmips_result_size_{fname_sufix}.pdf", bbox_inches='tight')


if __name__ == "__main__":
    dataset_l = ['Yelp']
    origin_dataset_l = [
        'yelp_weighted_query-150d-QSRPRefineComputeIPBound-top100-n_sample_7846-n_sample_query_5000-sample_topk_200-n_thread_2-single-query-performance.csv',
    ]
    # 'yelp-150d-top200-simpfer_k_max_300-n_cand.txt'
    xlim_l = [None]
    ylim_l = [None]
    x_ticks_l = [None]
    n_bin_l = [50]
    name_m = {'fig_x': '# Refine User',
              'fig_y': 'Frequency'}
    is_test = True
    for dataset_name_fig, dataset_name_file, xlim, ylim, x_ticks, n_bin in zip(dataset_l,
                                                                               origin_dataset_l, xlim_l,
                                                                               ylim_l, x_ticks_l,
                                                                               n_bin_l):
        df = pd.read_csv(dataset_name_file)
        # reuslt_size_l = np.where(reuslt_size_l < 1000)
        plot_figure(n_refine_user_l=df['n_refine_user'],
                    xlim=xlim, ylim=ylim, x_ticks=x_ticks, n_bin=n_bin,
                    fname_sufix=dataset_name_fig,
                    name_m=name_m, is_test=is_test)
