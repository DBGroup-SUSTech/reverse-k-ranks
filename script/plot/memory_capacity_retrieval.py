import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

linestyle_l = ['_', '-', '--', ':']
color_l = ['#3D0DFF', '#6BFF00', '#00E8E2', '#EB0225', '#FF9E03']
marker_l = ['x', "v", "o", "D", "s"]
markersize = 15

matplotlib.rcParams.update({'font.size': 20})


def plot_figure(*, fname: str, dataset_name: str, ylim: list,
                name_m: dict, method_m: dict, ylog: bool, legend_loc: list, test: bool):
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
        ax.plot(x_l, y_l,
                color='#000000', linewidth=2.5, linestyle='-',
                label=method_m[key],
                marker=marker_l[method_i], fillstyle='none', markersize=markersize)
    # ax.set_xlim([14, 280])
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlabel(name_m['fig_x'])
    ax.set_ylabel(name_m['fig_y'])
    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 4, 16, 64, 256])
    if ylog:
        ax.set_yscale('log', base=10)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax.legend(frameon=False, loc='best')
    ax.legend(frameon=False, loc=legend_loc[0], bbox_to_anchor=legend_loc[1])
    if test:
        plt.savefig("memory_capacity_retrieval_{}.jpg".format(dataset_name), bbox_inches='tight', dpi=600)
    else:
        plt.savefig("memory_capacity_retrieval_{}.pdf".format(dataset_name), bbox_inches='tight')


if __name__ == "__main__":
    is_test = False

    name_m = {'csv_x': 'memory_capacity', 'fig_x': 'Memory Capacity (GB)',
              'csv_y': 'RunningTime', 'fig_y': 'Query Time (Second)'}
    method_m = {'US': 'US', 'QSRP': 'QSRP'}
    fname_l = ['./data/memory_capacity_curve/Yelp-retrieval.csv',
               './data/memory_capacity_curve/Amazon-retrieval.csv', ]
    dataset_name_l = ['1_yelp', '2_amazon']
    # ylim_l = [None, None]
    ylog_l = [True, True]
    legend_loc_l = [('upper right', None), ('upper right', None)]
    ylim_l = [[1e-2, 8e0], [1e-2, 5e2]]
    # ylim_l = [None, None]
    for fname, dataset_name, ylim, ylog, legend_loc in zip(fname_l, dataset_name_l, ylim_l, ylog_l, legend_loc_l):
        plot_figure(fname=fname, dataset_name=dataset_name, ylim=ylim,
                    name_m=name_m, method_m=method_m, ylog=ylog, legend_loc=legend_loc, test=is_test)
