import matplotlib.pyplot as plt
import numpy as np

# first is n_user, next is n_data_item
from matplotlib.ticker import FuncFormatter

dataset_m ={
    'movielens-27m': [283228, 53889],
    'netflix': [480189, 17770],
}

# movielens
# n_user = 283228
# n_data_item = 53889
# fake-normal
# n_user = 1000
# n_data_item = 5000
# netflix
# n_user = 480189
# n_data_item = 17770
# yelp-small
# n_user = 500000
# n_data_item = 50000

if __name__ == '__main__':
    for top_perc in [1, 2, 5, 8, 10, 20, 40]:
        # dataset = 'movielens-27m'
        dataset = 'netflix'
        distribution_l = np.loadtxt(
            '../../result/attribution/norm-popularity-distribution-{}-perc-{}.csv'.format(dataset, top_perc),
            delimiter=',')
        scatter_x = distribution_l[:, 0]
        scatter_y = distribution_l[:, 1]

        # plot
        fig, ax = plt.subplots()

        # ax.scatter(x, y, s=sizes, c=colors, vmin=0, vmax=100)
        # ax.scatter(x, y, vmin=0, vmax=n_data_item)
        ax.scatter(scatter_x, scatter_y, s=2)

        # ax.set(xlim=(0, n_data_item),
        #        ylim=(0, n_data_item))
        # ax.set(xlim=(0, 5000),
        #        ylim=(0, 5000))

        ax.set_xlabel('query norm')
        ax.set_ylabel('# user in top {}% rank'.format(top_perc))

        from matplotlib import ticker
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1,1))
        ax.yaxis.set_major_formatter(formatter)

        # plt.ticklabel_format(axis='y', style='scientific', useMathText=True)

        # ax.set_title('{}'.format(dataset))

        # plt.show()
        plt.savefig('norm-popularity-distribution-{}-top-perc-{}.jpg'.format(dataset, top_perc))
        plt.close()
