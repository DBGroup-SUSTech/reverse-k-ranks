import matplotlib.pyplot as plt
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', '--dataset', help="dataset", dest='ds', type=str, required=True)
    args = parser.parse_args()
    ds = args.ds
    curve_l = np.loadtxt('../../result/attribution/{}-config.txt'.format(ds))
    x = np.arange(len(curve_l), dtype=np.float64) / len(curve_l)
    plt.plot(x, curve_l)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.xlabel("inner product percentile")
    plt.ylabel('probability accumulation')
    plt.title('{} distribution'.format(ds))
    plt.savefig('../../result/attribution/{}-distribution.jpg'.format(ds))
