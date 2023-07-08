import numpy as np
import matplotlib.pyplot as plt


def plot_scatter(data, fig_name):
    scatter_x = data[:, 0]
    scatter_y = data[:, 1]
    print("corr", similar_type, method_name, np.corrcoef(scatter_x, scatter_y)[0, 1], sampleID)

    # plot
    fig, ax = plt.subplots()

    # ax.scatter(x, y, s=sizes, c=colors, vmin=0, vmax=100)
    # ax.scatter(x, y, vmin=0, vmax=n_data_item)
    ax.scatter(scatter_x, scatter_y, s=2)

    # ax.set(xlim=(0, n_data_item),
    #        ylim=(0, n_data_item))
    # ax.set(xlim=(0, 5000),
    #        ylim=(0, 5000))

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('correlation {}'.format(np.corrcoef(scatter_x, scatter_y)[0, 1]))
    plt.savefig(fig_name)
    plt.close()


n_sample = 15
for dataset_name in ['movielens-27m', 'netflix']:
    for similar_type in ['far-user', 'nearest-user']:
        for method_name in ['single-rank-distribution', 'score-interval', 'compress-rank-relation']:
            for sampleID in range(n_sample):
                data = np.loadtxt(
                    '../../../result/attribution/SimilarItemRank/{}-{}-{}-sampleID-{}.csv'.format(
                        method_name, dataset_name, similar_type, sampleID),
                    delimiter=',')
                plot_scatter(data,
                             '../ignore-fig/correlation-{}-{}-{}-{}.jpg'.format(similar_type, method_name, sampleID,
                                                                                dataset_name))
