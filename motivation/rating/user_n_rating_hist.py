import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import json


def run(filename, dataset):
    rating_df = pd.read_csv(filename)
    rating_len_l = rating_df.groupby('userId').count()['rating'].to_numpy()
    # n_user = np.max(rating_df['userId'])
    # rating_len_l = [len(rating_df[rating_df['userId'] == uid]) for uid in range(1, n_user + 1, 1)]

    plt.hist(rating_len_l, bins=500, density=False, stacked=False)
    plt.savefig('result/user_n_rating/%s.jpg' % dataset)
    plt.close()

    with open('result/user_n_rating/%s.json' % dataset, 'w') as f:
        m = {
            'min_n_rating': str(np.min(rating_len_l)),
            'max_n_rating': str(np.max(rating_len_l)),
            'avg_n_rating': str(np.average(rating_len_l))
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
