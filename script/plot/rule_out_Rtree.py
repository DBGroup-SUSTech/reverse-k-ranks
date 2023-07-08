import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import ticker

linestyle_l = ['_', '-', '--', ':']
color_l = ['#3D0DFF', '#6BFF00', '#00E8E2', '#EB0225', '#FF9E03']
marker_l = ['x', "v", "o", "D", "s"]
markersize = 15

matplotlib.rcParams.update({'font.size': 20})


def plot_figure(*, fname: str, dataset_name: str, ylim: list,
                name_m: dict, method_m: dict, ylog: bool, legend_loc: list, dataset_info: list, test: bool):
    # fig = plt.figure(figsize=(25, 4))
    fig = plt.figure(figsize=(6, 4))
    subplot_str = 111
    ax = fig.add_subplot(subplot_str)
    df = pd.read_csv(fname)
    for method_i, key in enumerate(method_m.keys()):
        x_name = name_m['csv_x']
        y_name = key
        x_l = df[x_name]
        y_l = df[y_name]
        y_l = y_l / 1000
        y_l = y_l / ((dataset_m[dataset_name][0] + 1) * dataset_m[dataset_name][2])
        ax.plot(x_l, y_l,
                color='#000000', linewidth=2.5, linestyle='-',
                label=method_m[key],
                marker=marker_l[method_i], fillstyle='none', markersize=markersize)

    ax.set_xlabel(name_m['fig_x'])
    ax.set_ylabel(name_m['fig_y'])
    ax.set_xscale('log', base=2)
    ax.set_xticks([2, 8, 32, 128, 512])
    ax.set_xlim([1.5, 8e2])
    if ylog:
        ax.set_yscale('log', base=10)
    # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f%%'))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    # ax.set_yticks([0, 0.5, 1.0])
    if ylim:
        ax.set_ylim(ylim)
    # ax.legend(frameon=False, loc='best')
    ax.legend(frameon=False, loc=legend_loc[0], bbox_to_anchor=legend_loc[1])
    if test:
        plt.savefig("rule_out_Rtree_{}.jpg".format(dataset_name), bbox_inches='tight', dpi=600)
    else:
        plt.savefig("rule_out_Rtree_{}.pdf".format(dataset_name), bbox_inches='tight')


'''first is n_item, second is n_query, third is n_user'''
dataset_m = {'2_yelp': [159585, 1000, 2189457],
             '1_amazon': [409243, 1000, 2511610]}

if __name__ == "__main__":
    is_test = False

    name_m = {'csv_x': 'dimensionality', 'fig_x': 'Dimensionality',
              'csv_y': 'RunningTime', 'fig_y': 'Refinement Ratio'}
    method_m = {'Rtree': 'R-Tree', 'US': 'US', 'QSRP': 'QSRP'}
    fname_l = ['./data/rule_out_Rtree/Amazon.csv',
               './data/rule_out_Rtree/Yelp.csv']
    dataset_name_l = ['1_amazon', '2_yelp']
    # ylim_l = [None, None]
    ylog_l = [False, False]
    # legend_loc_l = [('center right', (1, 0.6)), ('center right', (1, 0.58))]
    legend_loc_l = [('best', None), ('best', None)]
    ylim_l = [[-0.1, 0.5], [-0.01, 0.1]]
    # ylim_l = [None, None]
    for fname, dataset_name, ylim, ylog, legend_loc in zip(fname_l, dataset_name_l, ylim_l, ylog_l, legend_loc_l):
        plot_figure(fname=fname, dataset_name=dataset_name, ylim=ylim,
                    name_m=name_m, method_m=method_m, ylog=ylog, legend_loc=legend_loc,
                    dataset_info=dataset_m[dataset_name],
                    test=is_test)
