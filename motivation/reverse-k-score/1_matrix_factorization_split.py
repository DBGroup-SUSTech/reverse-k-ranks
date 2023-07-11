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


'''used for generating the ml-1m data'''

def matrix_factorization_split(df: pd.DataFrame):
    userID_l = np.sort(np.unique(df['userID']))

    test_index_l = []
    # good_rating = np.median(df['rating'])
    good_rating = np.median(df['rating'])
    for userID in userID_l:
        item_freq = len(df[(df['userID'] == userID)])
        # index_value_l = df[(df['itemID'] == itemID) & (df['rating'] >= median_rating)].index.values
        # index_value_l = df[(df['userID'] == userID) & (df['rating'] < good_rating)].index.values
        index_value_l = df[(df['userID'] == userID) & (df['rating'] < good_rating)].index.values
        # index_value_l = df[(df['userID'] == userID)].index.values
        # if item_freq <= 100 and len(index_value_l > 0):
        if 0 < len(index_value_l) < 300:
            # index_i = np.random.randint(low=0, high=len(index_value_l), size=1)[0]
            # test_index_l.append(index_value_l[index_i])
            # index_i_l = np.arange(len(index_value_l))[:int(len(index_value_l) * 0.1)]
            n_sample_index = int(len(index_value_l) * 0.8) if int(len(index_value_l) * 0.8) > 1 else 1
            # n_sample_index = int(len(index_value_l) * 0.8) if int(len(index_value_l) * 0.8) > 1 else 1
            assert n_sample_index > 0
            for index_i in range(n_sample_index):
                test_index_l.append(index_value_l[index_i])

    # test_index_l = np.array(test_index_l)[:int(len(df) * 0.1)]
    np.random.seed(0)
    if len(test_index_l) > int(len(df) * 0.5):
        test_index_l = np.random.choice(test_index_l, int(len(df) * 0.5), replace=False)
    # test_index_l = np.array(test_index_l)[:int(len(df) * 0.5)]
    assert len(test_index_l) == len(np.unique(test_index_l))

    test_df = df.iloc[test_index_l]
    train_index_l = np.setdiff1d(np.arange(len(df)), test_index_l)
    # train_index_l = np.arange(len(df))
    train_df = df.iloc[train_index_l]
    assert len(test_df) + len(train_df) == len(df)
    print(len(test_df), len(train_df), len(df))
    return test_df, train_df


'''used for generating the lastfm data, previous'''


# def matrix_factorization_split(df: pd.DataFrame):
#     userID_l = np.sort(np.unique(df['userID']))
#
#     test_index_l = []
#     # good_rating = np.median(np.sort(np.unique(df['rating'])))
#     # upper_rating = np.median(df['rating']) * 1.1
#     # lower_rating = np.median(df['rating']) * 0
#
#     for userID in userID_l:
#         index_value_l = df[
#             (df['userID'] == userID)].index.values
#
#         for index in index_value_l:
#             itemID = df.iloc[index]['itemID']
#             rating = df.iloc[index]['rating']
#             rating_l = df[df['itemID'] == itemID]['rating'].to_numpy()
#             rank = np.sum(rating_l > rating)
#             # if 50 <= rank <= 60:
#             if 50 <= rank <= 55:
#             # if 40 <= rank <= 60:
#                 test_index_l.append(index)
#
#     for userID in userID_l:
#         index_value_l = df[
#             (df['userID'] == userID)].index.values
#
#         for index in index_value_l:
#             itemID = df.iloc[index]['itemID']
#             rating = df.iloc[index]['rating']
#             rating_l = df[df['itemID'] == itemID]['rating'].to_numpy()
#             rank = np.sum(rating_l > rating)
#             if 22 <= rank <= 35:
#             # if 16 <= rank <= 25:
#             # if 15 <= rank <= 20:
#                 test_index_l.append(index)
#
#     good_rating = np.median(df['rating'])
#     for userID in userID_l:
#         # index_value_l = df[(df['itemID'] == itemID) & (df['rating'] >= median_rating)].index.values
#         # index_value_l = df[(df['userID'] == userID) & (df['rating'] < good_rating)].index.values
#         index_value_l = df[(df['userID'] == userID) & (df['rating'] < good_rating)].index.values
#         # index_value_l = df[(df['userID'] == userID)].index.values
#         # if item_freq <= 100 and len(index_value_l > 0):
#         if 0 < len(index_value_l) < 2:
#             # index_i = np.random.randint(low=0, high=len(index_value_l), size=1)[0]
#             # test_index_l.append(index_value_l[index_i])
#             # index_i_l = np.arange(len(index_value_l))[:int(len(index_value_l) * 0.1)]
#             # n_sample_index = 1
#             n_sample_index = int(len(index_value_l) * 0.2) if int(len(index_value_l) * 0.2) > 1 else 1
#             assert n_sample_index > 0
#             for index_i in range(n_sample_index):
#                 test_index_l.append(index_value_l[index_i])
#
#     np.random.seed(0)
#     test_index_l = np.unique(test_index_l)
#     if len(test_index_l) > int(len(df) * 0.1):
#         test_index_l = np.random.choice(test_index_l, int(len(df) * 0.1), replace=False)
#
#     # test_index_l = np.array(test_index_l)[:int(len(df) * 0.2)]
#
#     assert len(test_index_l) == len(np.unique(test_index_l))
#
#     test_df = df.iloc[test_index_l]
#     train_index_l = np.setdiff1d(np.arange(len(df)), test_index_l)
#     # train_index_l = np.arange(len(df))
#     train_df = df.iloc[train_index_l]
#     assert len(test_df) + len(train_df) == len(df)
#     print(len(test_df), len(train_df), len(df))
#     return test_df, train_df


if __name__ == '__main__':
    # filename_l = ['lastfm']
    filename_l = ['ml-1m']
    # filename_l = ['lastfm', 'ml-1m']
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
