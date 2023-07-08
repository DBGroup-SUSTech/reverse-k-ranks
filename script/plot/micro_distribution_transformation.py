import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

linestyle_l = ['_', '-', '--', ':']
color_l = ['#3D0DFF', '#6BFF00', '#00E8E2', '#EB0225', '#FF9E03']
marker_l = ['x', "v", "o", "D", "s"]
markersize = 15

matplotlib.rcParams.update({'font.size': 20})


def plot_curve(*, x_l_l: list, y_l_l: list, label_l: list,
               xlabel: str, ylabel: str,
               ylim: list, log: bool,
               fname_suffix: str,
               is_test: bool):
    assert len(x_l_l) == len(y_l_l) == len(label_l)
    fig = plt.figure(figsize=(6, 4))
    subplot_str = int('111')
    ax1 = fig.add_subplot(subplot_str)
    ax1.plot(x_l_l[1].to_numpy(), (y_l_l[1] / 1000).to_numpy(),
             color='#000000', linewidth=2.5, linestyle='-',
             label=label_l[1],
             marker=marker_l[1], fillstyle='none', markersize=markersize)
    ax1.plot(x_l_l[0].to_numpy(), (y_l_l[0] / 1000).to_numpy(),
             color='#000000', linewidth=2.5, linestyle='-',
             label=label_l[0],
             marker=marker_l[0], fillstyle='none', markersize=markersize)


    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_ylim(ylim)
    ax1.set_xlim([0, 210])
    if log:
        ax1.set_yscale('log')
    ax1.legend(frameon=False, loc='best')
    if is_test:
        plt.savefig("micro_distribution_transformation_{}.jpg".format(fname_suffix), bbox_inches='tight', dpi=600)
    else:
        plt.savefig("micro_distribution_transformation_{}.pdf".format(fname_suffix), bbox_inches='tight')


if __name__ == "__main__":
    is_test = False
    label_l = ['QSRP', 'QSRP-DT']

    amazon_l = pd.read_csv('./data/micro_distribution_transformation/Amazon.csv')
    topk_l = amazon_l['topk']
    amazon_normal_l = amazon_l['QSRPNormalLP']
    amazon_uniform_l = amazon_l['QSRPUniformLP']

    x_l_l = [topk_l, topk_l]
    y_l_l = [amazon_normal_l, amazon_uniform_l]

    plot_curve(x_l_l=x_l_l, y_l_l=y_l_l, label_l=label_l,
               xlabel='k', ylabel='Query Time (Second)',
               ylim=[0, 0.14], log=False, is_test=is_test, fname_suffix='2_amazon')

    yelp_l = pd.read_csv('./data/micro_distribution_transformation/Yelp.csv')
    yelp_normal_l = yelp_l['QSRPNormalLP']
    yelp_uniform = yelp_l['QSRPUniformLP']

    x_l_l = [topk_l, topk_l]
    y_l_l = [yelp_normal_l, yelp_uniform]
    label_l = ['QSRP', 'QSRP-DT']

    plot_curve(x_l_l=x_l_l, y_l_l=y_l_l, label_l=label_l,
               xlabel='k', ylabel='Query Time (Second)',
               ylim=[0, 0.085], log=False, is_test=is_test, fname_suffix='1_yelp')
