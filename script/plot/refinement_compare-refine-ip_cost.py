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
        ax.plot(x_l.to_numpy(), y_l.to_numpy(),
                color='#000000', linewidth=2.5, linestyle='-',
                label=method_m[key],
                marker=marker_l[method_i], fillstyle='none', markersize=markersize)

    from matplotlib.ticker import ScalarFormatter
    xfmt = ScalarFormatter()
    xfmt.set_powerlimits((-3, 3))
    ax.get_yaxis().set_major_formatter(xfmt)
    # ax.get_yaxis().get_major_formatter().set_scientific(True)
    # ax.get_yaxis().get_major_formatter().set_powerlimits([0, 1e3])
    # ax.yaxis.major.formatter._useMathText = True
    # ax.get_yaxis().get_major_formatter().set_useMathText(True)
    # ax.get_yaxis().get_major_formatter().set_useOffset(1e5)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlabel(name_m['fig_x'])
    ax.set_ylabel(name_m['fig_y'])
    # ax.set_xscale('log', base=2)
    ax.set_xticks([0, 50, 100, 150, 200])
    if ylog:
        ax.set_yscale('log', base=10)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax.legend(frameon=False, loc='best')
    ax.legend(frameon=False, loc=legend_loc[0], bbox_to_anchor=legend_loc[1])
    if test:
        plt.savefig("refinement_compare-refine-ip-cost_{}.jpg".format(dataset_name), bbox_inches='tight', dpi=600)
    else:
        plt.savefig("refinement_compare-refine-ip-cost_{}.pdf".format(dataset_name), bbox_inches='tight')


if __name__ == "__main__":
    is_test = True

    name_m = {'csv_x': 'topk', 'fig_x': r'k',
              'csv_y': 'RunningTime', 'fig_y': 'Refine IP Cost'}
    method_m = {'ComputeAll': 'QSRP-ComputeAll', 'LEMP': 'QSRP-LEMP', 'ComputeIPBound': 'QSRP'}
    fname_l = ['./data/refinement_compare/Yelp-refine-IP-cost.csv',
               './data/refinement_compare/Amazon-refine-IP-cost.csv']
    dataset_name_l = ['1_yelp', '2_amazon']
    # ylim_l = [None, None]
    ylog_l = [False, False]
    legend_loc_l = [('upper left', (-0.01, 1.05)), ('upper left', (-0.01, 1.05))]
    # ylim_l = [[-1e4, 1.4e5], [-1e6, 2.1e7]]
    ylim_l = [[-1e4, 1.4e5], None]
    # ylim_l = [None, None]
    for fname, dataset_name, ylim, ylog, legend_loc in zip(fname_l, dataset_name_l, ylim_l, ylog_l, legend_loc_l):
        plot_figure(fname=fname, dataset_name=dataset_name, ylim=ylim,
                    name_m=name_m, method_m=method_m, ylog=ylog, legend_loc=legend_loc, test=is_test)
