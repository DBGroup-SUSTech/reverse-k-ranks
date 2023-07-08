import numpy as np
import matplotlib.pyplot as plt


def plot_rank_pdf(score_distri, idx, n_sample):
    # plot
    fig, ax = plt.subplots()

    n_data_item = len(score_distri)
    rank_pdf_x = np.arange(1, n_data_item + 1, 1)
    rank_pdf_y = score_distri

    # ax.scatter(x, y, s=sizes, c=colors, vmin=0, vmax=100)
    # ax.scatter(x, y, vmin=0, vmax=n_data_item)
    ax.plot(rank_pdf_x, rank_pdf_y, linestyle='solid', color='#000000', label="distribution")

    # get the highest score and lowest score
    high_score = score_distri[0]
    low_score = score_distri[n_data_item - 1]
    score_dist = (high_score - low_score) / (n_sample - 1)
    for itvID in range(n_sample):
        y_score = high_score - score_dist * itvID
        line_x = np.arange(-2000, n_data_item + 2000, 1)
        line_y = np.ones(n_data_item + 4000) * y_score
        if itvID == 0:
            ax.plot(line_x, line_y, linestyle='dashed', color='#b9529f', label="sampled by score")
        else:
            ax.plot(line_x, line_y, linestyle='dashed', color='#b9529f')

    high_rank = n_data_item
    low_rank = 1
    # rank_dist = int((high_rank - low_rank) // n_sample)
    rank_dist = 10000
    print("rank_dist", rank_dist)
    for sampleID in range(n_sample):
        x_rank = (sampleID + 1) * rank_dist
        line_x = np.ones(n_data_item + 4000) * x_rank
        line_y = np.arange(-2000, n_data_item + 2000, 1)
        if sampleID == 0:
            ax.plot(line_x, line_y, linestyle='dotted', color='#0084ff', label='sampled by rank')
        else:
            ax.plot(line_x, line_y, linestyle='dotted', color='#0084ff')

    ax.legend(loc='upper left')

    ax.set(xlim=(-2000, n_data_item + 2000),
           ylim=(score_distri[n_data_item - 1] - 2, score_distri[0] + 2))
    # ax.set(xlim=(0, 5000),
    #        ylim=(0, 5000))

    ax.set_xlabel('rank')
    ax.set_ylabel('score')
    # ax.set_title('movielens-27m, n_data_item={}'.format(n_data_item))

    # plt.show()
    plt.savefig('rank_pdf_{}.jpg'.format(idx), dpi=600, bbox_inches='tight')
    plt.close()


def plot_score_distribution(score_distri, idx):
    fig, ax = plt.subplots()

    ax.hist(score_distri, bins=16, linewidth=0.5, edgecolor="white")

    # ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
    #        ylim=(0, 56), yticks=np.linspace(0, 56, 9))
    ax.set_xlabel('score')
    ax.set_ylabel('frequency')
    ax.set_title('movielens-27m')

    # plt.show()
    plt.savefig('score_distribution_{}.jpg'.format(idx))
    plt.close()


def plot_all_score_distribution(score_distribution_l):
    fig, ax = plt.subplots()
    total_size = len(score_distribution_l) * len(score_distribution_l[0])
    all_score = score_distribution_l.reshape(total_size)

    ax.hist(all_score, bins=16, linewidth=0.5, edgecolor="white")

    # ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
    #        ylim=(0, 56), yticks=np.linspace(0, 56, 9))
    ax.set_xlabel('score')
    ax.set_ylabel('frequency')
    ax.set_title('movielens-27m')

    # plt.show()
    plt.savefig('all_score_distribution.jpg')
    plt.close()


if __name__ == '__main__':
    score_distribution_l = np.loadtxt(
        '/home/bianzheng/reverse-k-ranks/result/attribution/PrintUserRank/score-distribution-movielens-27m.csv',
        delimiter=',')
    plot_all_score_distribution(score_distribution_l)

    i = 9
    plot_rank_pdf(score_distribution_l[i], i, 5)
    plot_score_distribution(score_distribution_l[i], i)
