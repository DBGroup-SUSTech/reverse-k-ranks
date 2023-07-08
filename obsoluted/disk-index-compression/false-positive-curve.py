import numpy as np
import matplotlib.pyplot as plt


def plot(x_l, y_l):
    # plot
    fig, ax = plt.subplots()

    ax.plot(x_l, y_l, linestyle='solid', color='#000000', label="distribution")

    ax.legend(loc='upper left')

    # ax.set(xlim=(-2000, n_data_item + 2000),
    #        ylim=(score_distri[n_data_item - 1] - 2, score_distri[0] + 2))
    # ax.set(xlim=(0, 5000),
    #        ylim=(0, 5000))

    ax.set_xlabel('rank')
    ax.set_ylabel('score')
    # ax.set_title('movielens-27m, n_data_item={}'.format(n_data_item))

    # plt.show()
    plt.savefig('false-positive-curve.jpg'.format(1), dpi=600, bbox_inches='tight')
    plt.close()


def f(m, k, n):
    fp = (1 - (1 - 1 / m) ** (k * n)) ** k
    return fp


n_user = 6218834
n_data_item = 2166750

# m_l = np.arange(1, n_user, 1000)
m_l = np.array([100, 1000, 50000])
n = n_user
k_l = np.ceil(np.log(2) * n / m_l)
# k = 6
print(k_l)

fp_l = [f(m_l[i], k_l[i], n) for i in range(len(m_l))]
plot(m_l, fp_l)
