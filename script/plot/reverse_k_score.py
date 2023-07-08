import itertools

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import ticker

linestyle_l = ['_', '-', '--', ':']
color_l = ['#3D0DFF', '#6BFF00', '#00E8E2', '#EB0225', '#FF9E03']
marker_l = ['x', "v", "o", "D", "s"]
markersize = 15

matplotlib.rcParams.update({'font.size': 20})


def plot_figure(*, fname: str, dataset: str, set_log: bool, ylim: list, legend_loc: tuple, labelpad: int,
                name_m: dict, method_m: dict, yticks: list, test: bool):
    # fig = plt.figure(figsize=(25, 4))
    fig = plt.figure(figsize=(6, 4))
    # fig.text(label_pos_l[0], label_pos_l[1], name_m['fig_y'], va='center', rotation='vertical')
    subplot_str = 111
    ax = fig.add_subplot(subplot_str)
    df = pd.read_csv(fname)
    for method_i, key in enumerate(method_m.keys()):
        x_name = name_m['csv_x']
        y_name = key
        y_l = df[y_name]
        y_l = y_l

        ax.plot(df[x_name].to_numpy(), y_l.to_numpy(),
                color='#000000', linewidth=2.5, linestyle='-',
                label=method_m[key],
                marker=marker_l[method_i], fillstyle='none', markersize=markersize)
        print(x_name, key, dataset, y_l.to_numpy())

    ax.set_xlabel(name_m['fig_x'])
    ax.set_xlim([0, 210])
    ax.set_ylabel(name_m['fig_y'], labelpad=labelpad)
    if yticks:
        ax.set_yticks(yticks)
    if ylim:
        ax.set_ylim(ylim)
    if set_log:
        ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    handles, labels = plt.gca().get_legend_handles_labels()

    def flip(items, ncol):
        return itertools.chain(*[items[i::ncol] for i in range(ncol)])

    # ax.legend(reversed(handles), reversed(labels), frameon=False, ncol=2, loc=legend_loc[0], bbox_to_anchor=legend_loc[1])
    ax.legend(flip(handles, 2), flip(labels, 2), frameon=False, ncol=1, loc=legend_loc[0], bbox_to_anchor=legend_loc[1])
    if test:
        plt.savefig("reverse_k_score_{}.jpg".format(dataset), bbox_inches='tight', dpi=600)
    else:
        plt.savefig("reverse_k_score_{}.pdf".format(dataset), bbox_inches='tight')


if __name__ == "__main__":
    is_test = True

    fname_l = [
        './data/reverse-k-score/lastfm.csv',
        './data/reverse-k-score/ml-1m.csv',
    ]
    dataset_l = ['1_lastfm', '2_ml_1m']
    set_log_l = [False, False]
    ylim_l = [[-0.01, 0.17], [0, 0.07]]
    legend_loc_l = [('upper left', (0, 1)), ('upper left', (0, 1))]
    labelpad_l = [0, 0]
    yticks_l = [[0, 0.05, 0.10, 0.15], None]

    name_m = {'csv_x': 'topk', 'fig_x': r'k',
              'csv_y': 'Hitting Ratio', 'fig_y': 'Hitting Ratio'}
    method_m = {'krank': 'Reverse k-Rank', 'kscore': 'Reverse k-Score'}
    for fname, dataset, set_log, ylim, legend_loc, labelpad, yticks in zip(fname_l, dataset_l,
                                                                           set_log_l, ylim_l, legend_loc_l,
                                                                           labelpad_l, yticks_l):
        plot_figure(fname=fname, dataset=dataset, set_log=set_log, ylim=ylim, legend_loc=legend_loc, labelpad=labelpad,
                    name_m=name_m, method_m=method_m, yticks=yticks, test=is_test)
