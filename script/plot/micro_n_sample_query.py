import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

linestyle_l = ['_', '-', '--', ':']
marker_l = ['x', "v", "o", "D", "s"]
markersize = 15

hatch_l = ['//', '\\', '||', '++']
# style_l = [(None, '#ffffff'), ('\\', '#ffffff'), ('//', '#ffffff'), (None, '#000000')]
style_l = [(None, '#ffffff'), ('\\', '#ffffff'), (None, '#000000')]

matplotlib.rcParams.update({'font.size': 20,
                            # 'text.usetex': True,
                            # "font.family": "Helvetica"
                            })


def sample_query_curve(*, n_sample_query_l: list,
                       find_sample_rank_l: list,
                       build_score_table_l: list, compute_linear_regression_l: list,
                       query_time_top100_l: np.ndarray,
                       y1lim: list, ylim: list, legend_loc: list,
                       fname_sufix: str, is_test: bool):
    fig = plt.figure(figsize=(4 + 2, 4))
    subplot_str = int('111')
    ax1 = fig.add_subplot(subplot_str)
    ax2 = ax1.twinx()

    find_sample_rank_l = np.array(find_sample_rank_l) / 60
    build_score_table_l = np.array(build_score_table_l) / 60
    compute_linear_regression_l = np.array(compute_linear_regression_l) / 60

    ax1.bar(x=n_sample_query_l, height=find_sample_rank_l,
            label=r'Find Sample Rank', width=600,
            hatch=style_l[0][0], color=style_l[0][1], edgecolor='#000000')

    ax1.bar(x=n_sample_query_l, height=build_score_table_l,
            bottom=find_sample_rank_l,
            label='Build Score Table', width=600,
            hatch=style_l[1][0], color=style_l[1][1], edgecolor='#000000')

    ax1.bar(x=n_sample_query_l, height=compute_linear_regression_l,
            bottom=find_sample_rank_l + build_score_table_l,
            label='Build Regression Index', width=600,
            hatch=style_l[2][0], color=style_l[2][1], edgecolor='#000000')

    ax2.plot(n_sample_query_l, (query_time_top100_l / 1000).to_numpy(),
             color='#000000', linewidth=2.5, linestyle='-',
             label='Query Time',
             marker='v', fillstyle='none', markersize=markersize)

    ax1.set_xlabel('# Train Query')
    ax1.set_ylabel('Build Index Time (Minute)')
    ax1.set_ylim(y1lim)
    # ax1.set_yscale('log')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False,
               loc=legend_loc[0], bbox_to_anchor=legend_loc[1],
               labelspacing=0.2)
    # ax2.set_xticks([1000, 3000, 5000, 7000, 9000], [1000, 3000, 5000, 7000, 9000])
    ax2.set_xticks([10, 2500, 5000, 7500, 10000])

    ax2.set_ylabel('Query Time (Second)')
    # ax2.set_yscale('log')
    ax2.set_ylim(ylim)
    if is_test:
        plt.savefig("{}_{}.jpg".format('micro_n_sample_query', fname_sufix), bbox_inches='tight', dpi=600)
    else:
        plt.savefig("{}_{}.pdf".format('micro_n_sample_query', fname_sufix), bbox_inches='tight')


if __name__ == "__main__":
    is_test = False
    n_sample_query_l = [10, 2500, 5000, 7500, 10000]

    # yelp
    yelp_df = pd.read_csv('data/micro_n_sample_query/Yelp.csv')
    fname_sufix = '1_yelp'
    y1lim = [0, 30]
    ylim = [0, 0.21]
    # y1lim = None
    # ylim = None

    legend_loc = ['upper left', (0, 1.05)]
    sample_query_curve(n_sample_query_l=n_sample_query_l,
                       find_sample_rank_l=yelp_df['find_sample_rank'],
                       build_score_table_l=yelp_df['build_score_table'],
                       compute_linear_regression_l=yelp_df['compute_linear_regression'],
                       query_time_top100_l=yelp_df['query_top100'],
                       y1lim=y1lim, ylim=ylim, legend_loc=legend_loc,
                       fname_sufix=fname_sufix, is_test=is_test)

    # amazon
    amazon_df = pd.read_csv('data/micro_n_sample_query/Amazon.csv')
    fname_sufix = '2_amazon'
    y1lim = [0, 61]
    ylim = [0, 0.45]
    # y1lim = None
    # ylim = None

    legend_loc = ['upper left', (0, 1.05)]
    sample_query_curve(n_sample_query_l=n_sample_query_l,
                       find_sample_rank_l=amazon_df['find_sample_rank'],
                       build_score_table_l=amazon_df['build_score_table'],
                       compute_linear_regression_l=amazon_df['compute_linear_regression'],
                       query_time_top100_l=amazon_df['query_top100'],
                       y1lim=y1lim, ylim=ylim, legend_loc=legend_loc,
                       fname_sufix=fname_sufix, is_test=is_test)
