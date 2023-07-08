import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

linestyle_l = ['_', '-', '--', ':']
color_l = ['#3D0DFF', '#6BFF00', '#00E8E2', '#EB0225', '#FF9E03']
marker_l = ['x', "v", "o", "D", "s"]
markersize = 15

matplotlib.rcParams.update({'font.size': 20})


def plot_figure(*, fname: str, dataset_name: str, ylim: list, yticks: list,
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
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlabel(name_m['fig_x'])
    ax.set_ylabel(name_m['fig_y'])
    ax.set_xscale('log', base=2)
    ax.set_xticks([32, 64, 128, 256, 512])
    if yticks:
        ax.set_yticks(yticks)
    if ylog:
        ax.set_yscale('log', base=10)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax.legend(frameon=False, loc='best')
    ax.legend(frameon=False, loc=legend_loc[0], bbox_to_anchor=legend_loc[1]
              , labelspacing=0.2)
    if test:
        plt.savefig("dimensionality_scalability_{}.jpg".format(dataset_name), bbox_inches='tight', dpi=600)
    else:
        plt.savefig("dimensionality_scalability_{}.pdf".format(dataset_name), bbox_inches='tight')


if __name__ == "__main__":
    is_test = False

    name_m = {'csv_x': 'dimensionality', 'fig_x': 'Dimensionality',
              'csv_y': 'Dimensionality', 'fig_y': 'Query Time (Second)'}
    # method_m = {'Grid': 'Grid', 'RMIPS': 'RMIPS', 'US': 'US', 'QSRP': 'QSRP'}
    method_m = {'US': 'US', 'QSRP': 'QSRP'}
    fname_l = ['./data/dimensionality_scalability/Yelp.csv',
               './data/dimensionality_scalability/Amazon.csv', ]
    dataset_name_l = ['1_yelp', '2_amazon']
    # ylim_l = [None, None]
    ylog_l = [False, False]
    legend_loc_l = [('upper left', None), ('upper left', None)]
    # legend_loc_l = [('best', None), ('best', None)]
    ylim_l = [[0, 0.48], [0, 0.48]]
    # ylim_l = [None, None]
    # yticks_l = [None, [0, 0.05, 0.1]]
    yticks_l = [None, None]
    for fname, dataset_name, ylim, yticks, ylog, legend_loc in zip(fname_l, dataset_name_l, ylim_l, yticks_l, ylog_l,
                                                                   legend_loc_l):
        plot_figure(fname=fname, dataset_name=dataset_name, ylim=ylim,
                    name_m=name_m, method_m=method_m, yticks=yticks, ylog=ylog, legend_loc=legend_loc, test=is_test)
