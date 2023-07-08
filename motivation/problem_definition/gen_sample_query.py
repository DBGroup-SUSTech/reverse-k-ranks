import faiss
import numpy as np
import vecs_io
import pandas as pd
import os
import json


def delete_file_if_exist(dire):
    if os.path.exists(dire):
        command = 'rm -rf %s' % dire
        print(command)
        os.system(command)


def generate_popular_item(dataset, n_pop_item):
    base_dir = '/home/bianzheng/Dataset/Recommendation/'
    ratings_fname = os.path.join(base_dir, '%s-ratings.csv' % dataset)
    rating_l = pd.read_csv(ratings_fname)
    rating_l['oldMovieId'] = rating_l['oldMovieId'] - 1
    rating_l['itemId'] = rating_l['itemId'] - 1
    # 找到5分中最多的几个item
    rating_l = rating_l[rating_l['rating'] >= 5]
    rating_l = rating_l.groupby(['itemId', 'oldMovieId']).count()
    rating_l.reset_index(inplace=True, drop=False)
    del rating_l['userId']
    rating_l.rename(columns={'rating': 'rating_count'}, inplace=True)
    rating_l.sort_values('rating_count', ascending=False, inplace=True)
    rating_l = rating_l[:n_pop_item]

    title_fname = os.path.join(base_dir, 'raw-data', dataset, 'movies.csv')
    title_l = pd.read_csv(title_fname)
    title_l['movieId'] = title_l['movieId'] - 1
    merge_l = pd.merge(rating_l, title_l, how='inner', left_on=['oldMovieId'], right_on=['movieId'])
    del merge_l['oldMovieId']
    order = ['itemId', 'movieId', 'title', 'genres', 'rating_count']
    merge_l = merge_l[order]
    merge_l.sort_values('rating_count', ascending=False, inplace=True)
    return merge_l


def get_random_item_df(dataset, pop_idx_itemid_l, n_random_item):
    base_dir = '/home/bianzheng/Dataset/Recommendation'

    rating_fname = os.path.join(base_dir, '%s-ratings.csv' % dataset)
    rating_df = pd.read_csv(rating_fname)
    rating_df['oldMovieId'] = rating_df['oldMovieId'] - 1
    rating_df['itemId'] = rating_df['itemId'] - 1
    # print(n_total_item)
    rating_df = rating_df.groupby(['itemId', 'oldMovieId']).count()
    del rating_df['rating'], rating_df['userId']
    rating_df.reset_index(inplace=True, drop=False)
    rating_df.sort_values('itemId', ascending=True, inplace=True)
    rating_df.drop(labels=pop_idx_itemid_l, axis=0, inplace=True)
    random_df_idx_l = np.random.permutation(len(rating_df))[:n_random_item]
    rating_df = rating_df.iloc[random_df_idx_l].copy()

    title_fname = os.path.join(base_dir, 'raw-data', dataset, 'movies.csv')
    title_l = pd.read_csv(title_fname)
    title_l['movieId'] = title_l['movieId'] - 1
    merge_l = pd.merge(rating_df, title_l, how='inner', left_on=['oldMovieId'], right_on=['movieId'])
    del merge_l['oldMovieId']
    order = ['itemId', 'movieId', 'title', 'genres']
    merge_l = merge_l[order]
    merge_l.sort_values('itemId', ascending=False, inplace=True)
    return merge_l


if __name__ == '__main__':
    # reverse k ranks是给定item, 需要输出user
    dimension = 150
    n_pop_item = 100
    n_random_item = 1

    ds_l = ['movielens-small']
    for dataset in ds_l:
        input_dir = '/home/bianzheng/Dataset/MIPS/user_item'

        item_dir = os.path.join(input_dir, '%s-%dd' % (dataset, dimension), '%s_item.dvecs' % dataset)
        user_dir = os.path.join(input_dir, '%s-%dd' % (dataset, dimension), '%s_user.dvecs' % dataset)

        item_l, d = vecs_io.dvecs_read(item_dir)
        user_l, d = vecs_io.dvecs_read(user_dir)

        output_dir = '/home/bianzheng/Dataset/MIPS/Reverse-kRanks/problem-definition/%s' % dataset
        delete_file_if_exist(output_dir)
        os.mkdir(output_dir)

        pop_df = generate_popular_item(dataset, n_pop_item)
        pop_fname = os.path.join(output_dir, 'popular.csv')
        pop_df.to_csv(pop_fname, index=False)
        pop_idx_itemId_l = np.array(pop_df['itemId'].tolist())
        # print(pop_df)

        n_item = len(item_l)
        random_df = get_random_item_df(dataset, pop_idx_itemId_l, n_random_item)
        random_fname = os.path.join(output_dir, 'random.csv')
        random_df.to_csv(random_fname, index=False)
        random_idx_itemId_l = np.array(random_df['itemId'].tolist())

        print("n_pop_item %d, n_random_item %d" % (len(pop_idx_itemId_l), len(random_idx_itemId_l)))

        query_idx_itemId_l = np.append(pop_idx_itemId_l, random_idx_itemId_l)
        query_idx_movieId_l = np.append(np.array(pop_df['movieId'].tolist()), np.array(random_df['movieId'].tolist()))
        item_idx_l = np.arange(n_item)
        data_idx_l = np.sort(np.delete(item_idx_l, query_idx_itemId_l))

        np.savetxt('%s/query_item_pop_idx_l.txt' % output_dir, pop_idx_itemId_l, fmt="%d")
        np.savetxt('%s/query_item_random_idx_l.txt' % output_dir, random_idx_itemId_l, fmt="%d")

        query_item_csv = pd.DataFrame({'itemId': query_idx_itemId_l, 'movieId': query_idx_movieId_l})
        query_item_csv.to_csv('%s/query_item.csv' % output_dir, index=False)
        np.savetxt('%s/data_item.txt' % output_dir, data_idx_l, fmt="%d")

        query_item_l = item_l[query_idx_itemId_l, :]
        data_item_l = item_l[data_idx_l, :]

        vecs_io.dvecs_write("%s/%s_query_item.dvecs" % (output_dir, dataset), query_item_l)
        vecs_io.dvecs_write("%s/%s_data_item.dvecs" % (output_dir, dataset), data_item_l)
        vecs_io.dvecs_write("%s/%s_user.dvecs" % (output_dir, dataset), user_l)
        print("write %s complete" % dataset)
