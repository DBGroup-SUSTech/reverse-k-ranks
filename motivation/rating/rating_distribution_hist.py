import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import json


def run(filename, dataset):
    rating_df = pd.read_csv(filename)
    rating_l = rating_df['rating']

    plt.hist(rating_l, bins=20, density=False, stacked=False)
    plt.savefig('result/rating_distribution/%s.jpg' % dataset)
    plt.close()

    with open('result/rating_distribution/%s.json' % dataset, 'w') as f:
        m = {
            'min_n_rating': str(np.min(rating_l)),
            'max_n_rating': str(np.max(rating_l)),
            'avg_n_rating': str(np.average(rating_l))
        }
        json.dump(m, f)


# hitting rate 关于dimensionality的曲线
if __name__ == '__main__':
    filename_l = ['movielens-small', 'movielens-20m', 'movielens-25m', 'movielens-27m', 'netflix', 'amazon']
    # filename_l = ['movielens-small']
    base_dir = '/home/bianzheng/Dataset/Recommendation'
    # n_dim_l = np.arange(10, 100, 10)
    for fn in filename_l:
        csv_filename = os.path.join(base_dir, '%s-ratings.csv' % fn)
        run(csv_filename, fn)
        print("finish", fn)
