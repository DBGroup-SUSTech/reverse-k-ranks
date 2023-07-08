import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
import matplotlib
import math

linestyle_l = ['_', '-', '--', ':']
color_l = ['#3D0DFF', '#6BFF00', '#00E8E2', '#EB0225', '#FF9E03']
marker_l = ['x', "v", "o", "D", "s"]
markersize = 15

hatch = ['--', '+', 'x', '\\']
width = 0.35  # the width of the bars: can also be len(x) sequence


def plot_figure(*, dataset_l: list,
                fig_y: str, result_fname: str,
                prune_time_l: list, refine_time_l: list,
                is_test: bool):
    matplotlib.rcParams.update({'font.size': 30})
    n_dataset = len(dataset_l)
    # fig = plt.figure(figsize=(25, 4))
    fig = plt.figure(figsize=(10.5, 4.2))
    fig.text(0.01, 0.5, fig_y, va='center', rotation='vertical')

    subplot_str = int('111')
    ax = fig.add_subplot(subplot_str)

    prune_time_l = np.array(prune_time_l)
    refine_time_l = np.array(refine_time_l)

    total_time_l = prune_time_l + refine_time_l

    prune_time_perc_l = prune_time_l / total_time_l
    refine_time_perc_l = refine_time_l / total_time_l
    assert len(prune_time_l) == len(refine_time_l)

    width = 0.35
    position_l = np.sort(np.append(np.arange(n_dataset) - width / 28 * 18, np.arange(n_dataset) + width / 28 * 18))
    assert len(prune_time_l) == len(position_l)

    refine_time_ins = ax.bar(position_l, refine_time_perc_l, width - 0.1, color='#ffffff', edgecolor='#000000',
                             hatch='//', label='Refine Time')
    prune_time_ins = ax.bar(position_l, prune_time_perc_l, width - 0.1, bottom=refine_time_perc_l, color='#ffffff',
                            edgecolor='#000000',
                            hatch='', label='Prune Time')
    ax.set_ylim([0, 1.49])
    # ax.set_xlabel(dataset_l[fig_i])
    # ax.set_ylabel('')
    # ax.set_title(dataset_l[fig_i])
    # ax.legend(frameon=False, bbox_to_anchor=(0.5, 1), loc="center", ncol=len(dataset_l), borderaxespad=5)
    ax.legend(frameon=False, loc="upper center", ncol=len(dataset_l), borderaxespad=-0.2)
    # ax.set_xticks(np.arange(n_dataset), dataset_l)
    ax.set_xticks(np.arange(n_dataset))
    ax.set_xticklabels(dataset_l)
    ax.set_yticks([0, 0.5, 1.0])

    ax.bar_label(prune_time_ins, labels=['k=100', 'k=200', 'k=100', 'k=200'], padding=10)

    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    # ax.margins(y=50)

    fig.tight_layout(rect=(0.01, -0.11, 1.04, 1.12))
    # fig.tight_layout()
    if is_test:
        plt.savefig("{}.jpg".format(result_fname), dpi=600)
    else:
        plt.savefig("{}.pdf".format(result_fname))


if __name__ == "__main__":
    dataset_l = ['MovieLens', 'Amazon']
    fig_y = 'Pct.Query Time'
    result_fname = 'running_time_profile_QS'
    is_test = False

    # before advanced sampling solution
    prune_time_l = [15.757 + 50.400 + 8.712, 15.589 + 49.141 + 8.529,
                    22.362 + 50.172 + 8.368, 22.296 + 50.274 + 8.419]
    refine_time_l = [2.541, 4.583, 18.416, 24.230]

    plot_figure(dataset_l=dataset_l,
                fig_y=fig_y, result_fname=result_fname,
                prune_time_l=prune_time_l, refine_time_l=refine_time_l,
                is_test=is_test)
