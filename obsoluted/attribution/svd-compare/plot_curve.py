import matplotlib.pyplot as plt
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', '--dataset', help="dataset", dest='ds', type=str, required=True)
    args = parser.parse_args()
    ds = args.ds

    before_user_l = np.loadtxt('../../result/attribution/SVDCompare/%s-%s-distribution.txt' % (ds, "before_user"))
    before_item_l = np.loadtxt('../../result/attribution/SVDCompare/%s-%s-distribution.txt' % (ds, "before_item"))
    after_user_l = np.loadtxt('../../result/attribution/SVDCompare/%s-%s-distribution.txt' % (ds, "after_user"))
    after_item_l = np.loadtxt('../../result/attribution/SVDCompare/%s-%s-distribution.txt' % (ds, "after_item"))

    marker_l = ['H', 'D', 'P', '>', '*', 'X', 's', '<', '^', 'p', 'v']
    color_l = ['#b9529f', '#3953a4', '#ed2024', '#098140', '#231f20', '#7f8133', '#0084ff']

    plt.plot(before_user_l[:, 0], before_user_l[:, 1], marker=marker_l[0], linestyle='solid',
             color=color_l[0],
             label="before user")
    plt.plot(after_user_l[:, 0], after_user_l[:, 1], marker=marker_l[1], linestyle='dotted',
             color=color_l[1],
             label="after user")
    plt.savefig('./%s_user.png' % ds)
    plt.close()

    plt.plot(before_item_l[:, 0], before_item_l[:, 1], marker=marker_l[0], linestyle='solid',
             color=color_l[0],
             label="before user")
    plt.plot(after_item_l[:, 0], after_item_l[:, 1], marker=marker_l[1], linestyle='dotted',
             color=color_l[1],
             label="after user")
    plt.savefig('./%s_user.png' % ds)
