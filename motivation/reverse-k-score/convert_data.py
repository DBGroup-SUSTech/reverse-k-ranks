import numpy as np
import vecs_io
import multiprocessing
import time
import pandas as pd
import os


def delete_file_if_exist(dire):
    if os.path.exists(dire):
        command = 'rm -rf %s' % dire
        print(command)
        os.system(command)


def select_queryID_l(dataset_prefix: str, n_query_item: int):
    df = pd.read_csv(f'/home/bianzheng/rec2-mips/intermediate-rating-csv-split/{dataset_prefix}-test.csv')
    queryID_l = df['itemID'].to_numpy() - 1
    assert len(df) == len(np.unique(df['itemID']))
    return queryID_l


if __name__ == '__main__':
    # reverse k ranks是给定item, 需要输出user
    n_query_item = 100

    # ds_l = ['ml-1m']
    ds_l = ['lastfm', 'ml-1m']
    for dataset_prefix in ds_l:
        dim = 150
        dataset = f'{dataset_prefix}-{dim}d'
        input_dir = '/home/bianzheng/Dataset/MIPS'

        item_dir = os.path.join(input_dir, dataset, '%s_item.fvecs' % dataset_prefix)
        user_dir = os.path.join(input_dir, dataset, '%s_user.fvecs' % dataset_prefix)

        item_l, d = vecs_io.fvecs_read(item_dir)
        user_l, d = vecs_io.fvecs_read(user_dir)

        output_dir = '/home/bianzheng/Dataset/ReverseMIPS/%s' % dataset
        delete_file_if_exist(output_dir)
        os.mkdir(output_dir)

        n_item = len(item_l)
        # item_idx_l = np.random.permutation(n_item)
        # query_idx_l = np.sort(item_idx_l[:n_query_item])

        query_idx_l = np.array(select_queryID_l(dataset_prefix, n_query_item), dtype=np.int64)
        data_idx_l = np.arange(n_item)

        np.savetxt('%s/query_item.txt' % output_dir, query_idx_l, fmt="%d")
        np.savetxt('%s/data_item.txt' % output_dir, data_idx_l, fmt="%d")

        print(len(query_idx_l), len(item_l))
        query_item_l = item_l[query_idx_l, :]
        data_item_l = item_l

        vecs_io.fvecs_write("%s/%s_query_item.fvecs" % (output_dir, dataset), query_item_l)
        vecs_io.fvecs_write("%s/%s_data_item.fvecs" % (output_dir, dataset), data_item_l)
        vecs_io.fvecs_write("%s/%s_user.fvecs" % (output_dir, dataset), user_l)
        print("write %s complete" % dataset)
