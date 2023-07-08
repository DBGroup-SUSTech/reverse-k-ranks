import itertools

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

linestyle_l = ['_', '-', '--', ':']
color_l = ['#3D0DFF', '#6BFF00', '#00E8E2', '#EB0225', '#FF9E03']
marker_l = ['x', "v", "o", "D", "s"]
markersize = 15

matplotlib.rcParams.update({'font.size': 17})


def plot_figure(*, fname: str, dataset: str, set_log: bool, ylim: list, legend_loc: tuple, labelpad: int,
                name_m: dict, method_m: dict, result_fname_prefix: str, test: bool):
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
        y_l = y_l / 1000

        ax.plot(df[x_name].to_numpy(), y_l.to_numpy(),
                color='#000000', linewidth=2.5, linestyle='-',
                label=method_m[key],
                marker=marker_l[method_i], fillstyle='none', markersize=markersize)
        print(x_name, key, dataset, y_l.to_numpy())

    ax.set_xlabel(name_m['fig_x'])
    ax.set_xlim([0, 210])
    ax.set_ylabel(name_m['fig_y'], labelpad=labelpad)
    if ylim:
        ax.set_ylim(ylim)
    if set_log:
        ax.set_yscale('log')
    handles, labels = plt.gca().get_legend_handles_labels()

    def flip(items, ncol):
        return itertools.chain(*[items[i::ncol] for i in range(ncol)])

    # ax.legend(reversed(handles), reversed(labels), frameon=False, ncol=2, loc=legend_loc[0], bbox_to_anchor=legend_loc[1])
    ax.legend(flip(handles, 2), flip(labels, 2), frameon=False, ncol=2, loc=legend_loc[0], bbox_to_anchor=legend_loc[1])
    if test:
        plt.savefig("{}_{}.jpg".format(result_fname_prefix, dataset), bbox_inches='tight', dpi=600)
    else:
        plt.savefig("{}_{}.pdf".format(result_fname_prefix, dataset), bbox_inches='tight')


if __name__ == "__main__":
    is_test = False

    fname_l = [
        './data/k_curve/Yahoomusic-RunningTime.csv',
        './data/k_curve/Yelp-RunningTime.csv',
        './data/k_curve/Movielens-RunningTime.csv',
        './data/k_curve/Amazon-RunningTime.csv'
    ]
    dataset_l = ['1_Yahoomusic', '2_Yelp', '3_Movielens', '4_Amazon']
    set_log_l = [True, True, True, True]
    ylim_l = [[1e-2, 3e4], [1e-2, 5e4], [1e-2, 2e5], [1e-2, 2e5]]
    # ylim_l = [None, None, None, None]
    legend_loc_l = [('center right', (1, 0.4)), ('center right', (1, 0.4)), ('center right', (1, 0.4)),
                    ('center right', (1, 0.48))]
    # labelpad_l = [0, -5, -5, 0]
    labelpad_l = [0, 0, 0, 0]

    name_m = {'csv_x': 'topk', 'fig_x': r'k',
              'csv_y': 'RunningTime', 'fig_y': 'Query Time (Second)'}
    method_m = {'Grid': 'Grid', 'Rtree': 'Rtree', 'RMIPS': 'RMIPS', 'US': 'US', 'QSRP': 'QSRP'}
    result_fname_prefix = 'k_running_time'
    for fname, dataset, set_log, ylim, legend_loc, labelpad in zip(fname_l, dataset_l,
                                                                   set_log_l, ylim_l, legend_loc_l,
                                                                   labelpad_l):
        plot_figure(fname=fname, dataset=dataset, set_log=set_log, ylim=ylim, legend_loc=legend_loc, labelpad=labelpad,
                    name_m=name_m, method_m=method_m,
                    result_fname_prefix=result_fname_prefix, test=is_test)
