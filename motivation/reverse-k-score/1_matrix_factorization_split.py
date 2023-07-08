import numpy as np
import os
import pandas as pd
from libpmf.python import libpmf
import vecs_io
import copy


def delete_dir_if_exist(dire):
    if os.path.isdir(dire):
        command = 'rm -rf %s' % dire
        print(command)
        os.system(command)


def matrix_factorization_split(df: pd.DataFrame):
    itemID_l = np.sort(np.unique(df['itemID']))

    test_index_l = []
    median_rating = np.median(np.sort(np.unique(df['rating'])))
    for itemID in itemID_l:
        item_freq = len(df[(df['itemID'] == itemID)])
        # index_value_l = df[(df['itemID'] == itemID) & (df['rating'] >= median_rating)].index.values
        index_value_l = df[(df['itemID'] == itemID)].index.values
        # if item_freq <= 100 and len(index_value_l > 0):
        if len(index_value_l > 0):
            index_i = np.random.randint(low=0, high=len(index_value_l), size=1)[0]
            # index_i = 0
            test_index_l.append(index_value_l[index_i])
    test_index_l = np.array(test_index_l)

    test_df = df.iloc[test_index_l]
    train_index_l = np.setdiff1d(np.arange(len(df)), test_index_l)
    train_df = df.iloc[train_index_l]
    assert len(test_df) + len(train_df) == len(df)
    print(len(test_df), len(train_df), len(df))
    return test_df, train_df


if __name__ == '__main__':
    # filename_l = ['ml-1m']
    filename_l = ['lastfm', 'ml-1m']
    n_dim = 150
    username = 'bianzheng'

    for filename in filename_l:
        input_dir = f'/home/{username}/rec2-mips/intermediate-rating-csv'
        output_dir = f'/home/{username}/rec2-mips/intermediate-rating-csv-split'

        csv = pd.read_csv(os.path.join(input_dir, '%s.csv' % filename))
        test_df, train_df = matrix_factorization_split(df=csv)
        test_df.to_csv(os.path.join(output_dir, f'{filename}-test.csv'), index=False)
        train_df.to_csv(os.path.join(output_dir, f'{filename}-train.csv'), index=False)

    for filename in filename_l:
        input_dir = f'/home/{username}/rec2-mips/intermediate-rating-csv-split'

        csv = pd.read_csv(os.path.join(input_dir, f'{filename}-train.csv'))
        test_csv = pd.read_csv(os.path.join(input_dir, f'{filename}-test.csv'))
        n_user = np.max([np.max(csv['userID']), np.max(test_csv['userID'])])
        n_item = np.max([np.max(csv['itemID']), np.max(test_csv['itemID'])])
        res = libpmf.train_coo(row_idx=csv['userID'] - 1, col_idx=csv['itemID'] - 1, obs_val=csv['rating'],
                               m=n_user,
                               n=n_item, param_str='-k %d -n 20' % n_dim)
        user_l = res['W']
        item_l = res['H']

        save_dir = f'/home/{username}/Dataset/MIPS'
        save_filename = os.path.join(save_dir, '%s-%dd' % (filename, n_dim))
        delete_dir_if_exist(save_filename)
        os.mkdir(save_filename)

        save_user = os.path.join(save_filename, '%s_user.fvecs' % filename)
        save_item = os.path.join(save_filename, '%s_item.fvecs' % filename)
        vecs_io.fvecs_write(save_user, user_l)
        vecs_io.fvecs_write(save_item, item_l)
        print(filename, n_dim, 'complete n_user {}, n_item {}'.format(n_user, n_item))
