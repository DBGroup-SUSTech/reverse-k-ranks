import itertools

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

linestyle_l = ['_', '-', '--', ':']
# color_l = ['#3D0DFF', '#6BFF00', '#00E8E2', '#EB0225', '#FF9E03']
marker_l = ['x', "v", "o", "D", "s"]
markersize = 15
hatch_l = ['//', '\\', '||', '++']
color_l = ['#ffffff', '#ffffff', '#000000']
style_l = [(None, '#ffffff'), ('\\', '#ffffff'), ('//', '#ffffff'), ('++', '#ffffff'), (None, '#000000')]

matplotlib.rcParams.update({'font.size': 15})


def plot_figure(*, fname: str, dataset: str, set_log: bool, ylim: list, labelpad: int,
                name_m: dict, method_m: dict, result_fname_prefix: str, is_test: bool):
    # fig = plt.figure(figsize=(25, 4))
    fig = plt.figure(figsize=(6, 4))
    subplot_str = 111
    ax = fig.add_subplot(subplot_str)
    df = pd.read_csv(fname)
    topk_l = [int(topk) for topk in df['topk']]

    width = 0.25 if len(method_m.keys()) != 5 else 0.15

    offset = width / 2 if len(method_m.keys()) == 2 else width if len(method_m.keys()) == 3 else 1.5 * width

    # offset = width / len(method_m.keys())
    rects_l = []

    for method_i, key in enumerate(method_m.keys()):
        y_name = key
        x_l = np.arange(len(topk_l))
        y_l = df[y_name]
        y_l = y_l / 1000
        if len(method_m.keys()) == 2:
            rects = ax.bar(x_l + offset - method_i * width,
                           y_l, width,
                           color=style_l[method_i + 1][1], edgecolor='#000000',
                           hatch=style_l[method_i + 1][0], label=method_m[key])
        elif len(method_m.keys()) == 3:
            rects = ax.bar(x_l + offset - method_i * width,
                           y_l, width,
                           color=style_l[method_i][1], edgecolor='#000000',
                           hatch=style_l[method_i][0], label=method_m[key])
        elif len(method_m.keys()) == 4:
            rects = ax.bar(x_l + offset - method_i * width,
                           y_l, width,
                           color=style_l[method_i][1], edgecolor='#000000',
                           hatch=style_l[method_i][0], label=method_m[key])
        elif len(method_m.keys()) == 5:
            rects = ax.bar(x_l + offset - method_i * width,
                           y_l, width,
                           color=style_l[method_i][1], edgecolor='#000000',
                           hatch=style_l[method_i][0], label=method_m[key])
        rects_l.append(rects)

    ax.set_ylim(ylim)
    # ax.set_xlabel(dataset_l[fig_i])
    ax.set_ylabel(name_m['fig_y'], labelpad=labelpad)
    if set_log:
        ax.set_yscale('log')
    # ax.set_title(dataset_l[fig_i])
    # ax.legend(frameon=False, bbox_to_anchor=(0.5, 1), loc="center", ncol=len(dataset_l), borderaxespad=5)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.reverse()
    labels.reverse()

    def flip(items, ncol):
        return itertools.chain(*[items[i::ncol] for i in range(ncol)])

    ax.legend(flip(handles, 3), flip(labels, 3), frameon=False, loc="upper center", ncol=3, borderaxespad=-0,
              # columnspacing=0.8, handletextpad=0.1, handlelength=1.5, labelspacing=0.1)
              columnspacing=0.8, handletextpad=0.4)
    # ax.set_xticks(np.arange(n_dataset), dataset_l)
    # ax.set_yticks([0, 0.5, 1.0])
    x_name = name_m['csv_x']
    # ax.set_xticks(np.arange(len(topk_l)), ['{:.0f}'.format(topk) for topk in df[x_name]])
    ax.set_xticks(np.arange(len(topk_l)))
    ax.set_xticklabels(['{:.0f}'.format(topk) for topk in df[x_name]])
    ax.set_xlabel('k')

    # ax.bar_label(io_time_ins, labels=['Top-10', 'Top-100', 'Top-10', 'Top-100'], padding=7)
    # ax.margins(y=50)
    # fig.tight_layout()
    if is_test:
        plt.savefig("{}_{}.jpg".format(result_fname_prefix, dataset), bbox_inches='tight', dpi=600)
    else:
        plt.savefig("{}_{}.pdf".format(result_fname_prefix, dataset), bbox_inches='tight')


if __name__ == "__main__":
    is_test = False

    fname_l = ['./data/k_curve/Yahoomusic-IPCost.csv',
               './data/k_curve/Yelp-IPCost.csv',
               './data/k_curve/Movielens-IPCost.csv',
               './data/k_curve/Amazon-IPCost.csv']
    dataset_l = ['1_Yahoomusic', '2_Yelp', '3_Movielens', '4_Amazon']
    set_log_l = [True, True, True, True]
    ylim_l = [[1e5, 7e11], [1e5, 7e11], [1e5, 9e12], [1e5, 8e12]]
    # ylim_l = [None, None, None, None]
    labelpad_l = [0, 0, 0, 0]

    name_m = {'csv_x': 'topk', 'fig_x': r'k',
              'csv_y': 'IPCost', 'fig_y': '# Score Computation'}
    method_m = {'QSRP': 'QSRP', 'US': 'US', 'RMIPS': 'RMIPS', 'Rtree': 'Rtree', 'Grid': 'Grid'}
    result_fname_prefix = 'k_ip_cost'
    for fname, dataset, set_log, ylim, labelpad in zip(fname_l, dataset_l,
                                                       set_log_l, ylim_l,
                                                       labelpad_l):
        plot_figure(fname=fname, dataset=dataset, set_log=set_log, ylim=ylim, labelpad=labelpad,
                    name_m=name_m, method_m=method_m,
                    result_fname_prefix=result_fname_prefix, is_test=is_test)
