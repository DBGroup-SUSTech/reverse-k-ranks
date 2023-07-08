import os.path
import numpy as np

import pandas as pd

if __name__ == '__main__':
    topk_l = [10, 20, 30, 40, 50, 100, 200, 300, 400]
    for topk in topk_l:
        fname = os.path.join("../result", 'movielens-27m-QueryRankSample-top{}-n_sample_104-userID.csv'.format(topk))
        df = pd.read_csv(fname)
        avg_io_cost = np.average(df['io_cost'])
        print("topk {}, io_cost {}".format(topk, avg_io_cost))
