import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

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


def plot(data_x, data_y, x_axis_name, y_axis_name, title_name, file_name):
    print("corr", np.corrcoef(data_x, data_y))

    # plot
    fig, ax = plt.subplots()

    # ax.scatter(x, y, s=sizes, c=colors, vmin=0, vmax=100)
    # ax.scatter(x, y, vmin=0, vmax=n_data_item)
    max_num = max(max(data_x), max(data_y))

    arr = np.linspace(0, max_num, 100)
    ax.plot(arr, arr, color='#828487', linestyle='dotted')
    ax.scatter(data_x, data_y, s=2, color='#000000')

    # ax.set(xlim=(0, n_data_item),
    #        ylim=(0, n_data_item))
    # ax.set(xlim=(0, 5000),
    #        ylim=(0, 5000))

    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    ax.set_xlabel(x_axis_name)
    ax.set_ylabel(y_axis_name)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_title(title_name)

    plt.savefig('{}.jpg'.format(file_name), dpi=600, bbox_inches='tight')
    # plt.savefig('{}.pdf'.format(file_name), bbox_inches='tight')
    plt.close()


def least_square_parameter(x, y):
    log_x = x
    log_y = y
    # log_x = np.log10(x)
    # log_y = np.log10(y)
    # assemble matrix A
    A = np.vstack([log_x, np.ones(len(log_x))]).T

    # turn log_y into a column vector
    log_y = np.array(log_y)
    log_y = log_y[:, np.newaxis]
    # Direct least square regression
    alpha = np.dot((np.dot(np.linalg.inv(np.dot(A.T, A)), A.T)), log_y)
    print(alpha)
    return alpha


if __name__ == '__main__':
    # dataset_m = {'movielens-27m': 'Movielens', 'netflix': 'Netflix', 'yahoomusic_big': 'Yahoomusic'}
    dataset_m = {'yahoomusic_big': 'Yahoomusic', 'yelp': 'Yelp'}
    dataset_info_m = {'movielens-27m': [283228, 53889], 'netflix': [480189, 17770], 'yahoomusic_big': [1823179, 135736]}
    # dataset_l = ['movielens-27m', 'netflix', 'yahoomusic_big']
    for ds in dataset_m.keys():
        basic_dir = "../../result/laptop/single_query_performance"
        fname = "{}-RSTopTIP-top10-n_sample_1500-index_size_gb_256-userID.csv".format(ds)
        df_rstoptip = pd.read_csv(os.path.join(basic_dir, fname),
                                  dtype={'queryID': np.int64, "n_user_candidate": np.int64, "io_cost": np.int64,
                                         "ip_cost": np.int64,
                                         "total_time": np.float32, "io_time": np.float32, "ip_time": np.float32,
                                         "memory_index_time": np.float32},
                                  skipinitialspace=True)

        fname = "{}-QRSTopTIP-top10-n_sample_1500-index_size_gb_256-userID.csv".format(ds)
        df_qrstoptip = pd.read_csv(os.path.join(basic_dir, fname),
                                   dtype={'queryID': np.int64, "n_user_candidate": np.int64, "io_cost": np.int64,
                                          "ip_cost": np.int64,
                                          "total_time": np.float32, "io_time": np.float32, "ip_time": np.float32,
                                          "memory_index_time": np.float32},
                                   skipinitialspace=True)

        rs_n_user_candidate = df_rstoptip['n_user_candidate']
        qrs_n_user_candidate = df_qrstoptip['n_user_candidate']
        plot(rs_n_user_candidate, qrs_n_user_candidate, "Rank Sample", 'Query Rank Sample',
             '# Refinement Comparison in {}'.format(dataset_m[ds]),
             '{}-Refinement-Comparison'.format(ds))
