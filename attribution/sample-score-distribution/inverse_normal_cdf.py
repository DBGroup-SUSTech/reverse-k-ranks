from scipy.stats import norm
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
linestyle_l = ['_', '-', '--', ':']
color_l = ['#3D0DFF', '#6BFF00', '#00E8E2', '#EB0225', '#FF9E03']
marker_l = ['H', 'D', 'P', '>', '*', 'X', 's', '<', '^', 'p', 'v']
markersize = 10
matplotlib.RcParams.update(params)


def plot(a_l, n_point, fname):
    for i, a in enumerate(a_l, 0):
        x_l = [val / n_point * 5 - 2.5 for val in range(n_point)]
        y_l = [-norm.cdf(x) * a for x in x_l]
        print(x_l[1], a)
        plt.plot(x_l, y_l,
                 color=color_l[i], linewidth=0.5, linestyle='-',
                 label='para: {}'.format(a),
                 marker='H', markersize=0.5)
    plt.xlabel('sampled rank')
    # ax.set_ylabel('Running Time (ms)')
    # ax.set_ylim(0)
    # plt.yscale('log')
    plt.ylabel('IP')
    plt.legend()
    plt.savefig(fname, dpi=600)
    plt.close()


n_point = 1000
para_l = [5, 4, 3, 2, 1]
plot(para_l, n_point, "inverse-normal-cdf.jpg")
