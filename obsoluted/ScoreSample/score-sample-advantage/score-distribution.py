import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats

params = {
    'axes.labelsize': 8,
    'font.size': 8,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': False,
    'figure.figsize': [4.5, 4.5]
}
matplotlib.RcParams.update(params)


def plot_rank_pdf(n_item, n_sample):
    # plot
    mu = 0
    sigma = 1
    item_x_l = np.linspace(mu - 3 * sigma, mu + 3 * sigma, n_item)
    item_y_l = np.array([stats.norm.pdf(_, mu, sigma) for _ in item_x_l])

    fig, axs = plt.subplots(2, sharex=True)
    axs[0].plot(item_x_l, item_y_l, color='#000000')
    axs[0].set_title('Sample by Rank')

    axs[1].plot(item_x_l, item_y_l, color='#000000')
    axs[1].set_title('Sample by Score')
    n_point = 100

    high_rank = mu + 3 * sigma
    low_rank = mu - 3 * sigma
    total_n_item = stats.norm.cdf(high_rank) - stats.norm.cdf(low_rank)
    sample_n_item = total_n_item / (n_sample - 1)
    for i in range(n_sample):
        x_perc = stats.norm.ppf(stats.norm.cdf(low_rank) + sample_n_item * i)

        line_x = np.ones(n_point) * x_perc
        line_y = np.linspace(-0, 0.4, n_point)
        if i == 0:
            axs[0].plot(line_x, line_y, linestyle='dashed', color='#828487', label="sampled by score")
        else:
            axs[0].plot(line_x, line_y, linestyle='dashed', color='#828487')

    high_score = item_x_l[0]
    low_score = item_x_l[n_item - 1]
    score_dist = (high_score - low_score) / (n_sample - 1)
    for itvID in range(n_sample):
        x_score = high_score - score_dist * itvID
        line_x = np.ones(n_point) * x_score
        line_y = np.linspace(-0, 0.4, n_point)
        if itvID == 0:
            axs[1].plot(line_x, line_y, linestyle='dashed', color='#828487', label="sampled by score")
        else:
            axs[1].plot(line_x, line_y, linestyle='dashed', color='#828487')

    # ax.set(xlim=(0, 5000),
    #        ylim=(0, 5000))
    # ax.set_title('movielens-27m, n_data_item={}'.format(n_data_item))
    fig.tight_layout()
    axs[0].get_xaxis().set_visible(False)
    axs[0].get_yaxis().set_visible(False)
    axs[1].get_xaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)
    # plt.show()
    # plt.savefig('rank_pdf.jpg', dpi=600, bbox_inches='tight')
    plt.savefig('ScoreSampleAdvantage.pdf')
    plt.close()


plot_rank_pdf(1000, 10)
