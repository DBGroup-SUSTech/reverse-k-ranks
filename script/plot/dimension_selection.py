import json
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os

linestyle_l = ['_', '-', '--', ':']
color_l = ['#3D0DFF', '#6BFF00', '#00E8E2', '#EB0225', '#FF9E03']
marker_l = ['D', "v", "o", "x", "s"]
markersize = 15

matplotlib.rcParams.update({'font.size': 15})


def transform_data(*, dataset: str, dim_l: list, base_dir: str):
    hit_50_l = []
    hit_100_l = []
    hit_200_l = []
    for dim in dim_l:
        with open(os.path.join(base_dir, 'hitting_rate-{}-{}-new.json'.format(dataset, dim)),
                  'r') as f:
            json_ins = json.load(f)
        hit_50_l.append(json_ins['test_result']['hit@50'])
        hit_100_l.append(json_ins['test_result']['hit@100'])
        hit_200_l.append(json_ins['test_result']['hit@200'])
    df = pd.DataFrame({'dimension': dim_l, 'HR@50': hit_50_l, 'HR@100': hit_100_l, 'HR@200': hit_200_l})
    return df


def plot_figure(*, dataset_name: str, result_fname: str, base_dir: str,
                name_m: dict, method_m: dict, ylim: list, dim_l: list, legend_pos: list,
                xlog: bool, test: bool):
    # fig = plt.figure(figsize=(25, 4))
    fig = plt.figure(figsize=(6, 4))
    subplot_str = 111
    ax = fig.add_subplot(subplot_str)
    df = transform_data(dataset=dataset_name, dim_l=dim_l, base_dir=base_dir)
    for method_i, key in enumerate(method_m.keys()):
        x_name = name_m['csv_x']
        y_name = key + name_m['csv_y']
        ax.plot(df[x_name], df[y_name],
                color='#000000', linewidth=2.5, linestyle='-',
                label=method_m[key],
                marker=marker_l[method_i], fillstyle='none', markersize=markersize)
    ax.set_ylim(ylim)

    ax.set_xlabel(name_m['fig_x'])
    ax.set_ylabel(name_m['fig_y'])
    if legend_pos:
        ax.legend(frameon=False, loc=legend_pos[0], borderaxespad=legend_pos[1])
    else:
        ax.legend(frameon=False, loc='best', borderaxespad=-0)
    ax.set_xticks(dim_l)
    if xlog:
        ax.set_xscale('log', base=2)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # fig.tight_layout(rect=(0, 0.1, 1, 1))
    if test:
        plt.savefig("{}_{}.jpg".format('DimensionSelection', result_fname), bbox_inches='tight', dpi=600)
    else:
        plt.savefig("{}_{}.pdf".format('DimensionSelection', result_fname), bbox_inches='tight')


if __name__ == "__main__":
    is_test = False

    dataset_name_l = ['lastfm', 'ml-1m']
    ylim_l = [[0.19, 0.63], [0.19, 0.79]]
    name_m = {'csv_x': 'dimension', 'fig_x': 'Dimensionality',
              'csv_y': '', 'fig_y': 'Hitting Ratio'}
    method_m = {'HR@200': 'HR@200', 'HR@100': 'HR@100', 'HR@50': 'HR@50'}

    dim_l = [8, 16, 32, 64, 128, 256]
    base_dir = './data/dimension_selection/raw_data/2base'
    result_fname_l = ['1_lastfm', '2_movielens-1m']
    legend_pos_l = [('lower right', 0), None]
    for dataset_name, result_fname, ylim, legend_pos in zip(dataset_name_l, result_fname_l, ylim_l, legend_pos_l):
        plot_figure(dataset_name=dataset_name, result_fname=result_fname,
                    base_dir=base_dir,
                    name_m=name_m, method_m=method_m,
                    ylim=ylim, dim_l=dim_l, legend_pos=legend_pos,
                    xlog=True, test=is_test)
