import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

linestyle_l = ['_', '-', '--', ':']
color_l = ['#3D0DFF', '#6BFF00', '#00E8E2', '#EB0225', '#FF9E03']
marker_l = ['x', "v", "o", "D", "s"]
markersize = 15

matplotlib.rcParams.update({'font.size': 20})


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
        y_name = key + name_m['csv_y']
        ax.plot(list(df[x_name]), list(df[y_name] / 1000),
                color='#000000', linewidth=2.5, linestyle='-',
                label=method_m[key],
                marker=marker_l[method_i], fillstyle='none', markersize=markersize)

    ax.set_xlabel(name_m['fig_x'])
    # ax.set_xlim([0, 420])
    ax.set_xticks([0, 1, 2, 3, 4])
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[0] = '0%'
    labels[1] = '0.5%'
    labels[2] = '1.0%'
    labels[3] = '1.5%'
    labels[4] = '2.0%'
    ax.set_xticklabels(labels)

    # ax.set_xticks([10, 100, 200, 300, 400])
    ax.set_ylabel(name_m['fig_y'], labelpad=labelpad)
    if ylim:
        ax.set_ylim(ylim)
    if set_log:
        ax.set_yscale('log')
    ax.legend(frameon=False, loc=legend_loc[0], bbox_to_anchor=legend_loc[1])
    if test:
        plt.savefig("{}_{}.jpg".format(result_fname_prefix, dataset), bbox_inches='tight', dpi=600)
    else:
        plt.savefig("{}_{}.pdf".format(result_fname_prefix, dataset), bbox_inches='tight')


if __name__ == "__main__":
    is_test = True

    fname_l = ['./data/micro_index_update/Yelp.csv',
               './data/micro_index_update/Amazon.csv']
    dataset_l = ['1_Yelp', '2_Amazon']
    # set_log_l = [True, True]
    set_log_l = [False, False]
    ylim_l = [[0, 0.14], [0, 0.22]]
    # ylim_l = [None, None]
    legend_loc_l = [('lower right', None), ('lower right', None)]
    labelpad_l = [0, 0]

    name_m = {'csv_x': 'updateID', 'fig_x': r'Product Update Percentage',
              'csv_y': '', 'fig_y': 'Query Time (Second)'}
    method_m = {'insert': 'Insert', 'delete': 'Delete'}
    result_fname_prefix = 'micro_index_update'
    for fname, dataset, set_log, ylim, legend_loc, labelpad in zip(fname_l, dataset_l,
                                                                   set_log_l, ylim_l, legend_loc_l,
                                                                   labelpad_l):
        plot_figure(fname=fname, dataset=dataset, set_log=set_log, ylim=ylim, legend_loc=legend_loc, labelpad=labelpad,
                    name_m=name_m, method_m=method_m,
                    result_fname_prefix=result_fname_prefix, test=is_test)
