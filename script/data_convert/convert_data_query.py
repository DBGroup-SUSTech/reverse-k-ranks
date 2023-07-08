import numpy as np
import os
import vecs_io


def delete_file_if_exist(dire):
    if os.path.exists(dire):
        command = 'rm -rf %s' % dire
        print(command)
        os.system(command)


def get_sample_queryID(file_name):
    itemID_l = np.loadtxt(
        '/home/zhengbian/reverse-k-ranks/index/query_distribution/{}/sample_itemID_l.txt'.format(file_name))
    n_sample_item = len(itemID_l)
    sample_itemID = itemID_l[0]
    return sample_itemID


if __name__ == '__main__':
    ds_m = {'amazon-home-kitchen-150d': 'amazon-home-kitchen-150d'}
    # basic_dir = '/home/bianzheng/Dataset/ReverseMIPS'
    basic_dir = '/home/zhengbian/Dataset/ReverseMIPS'
    # basic_dir = os.path.join('/run', 'media', 'hdd', 'ReverseMIPS')
    # n_user = 1000
    # n_item = 5000
    n_query = 1000
    replace_queryID = 435
    n_last_query = 300

    for from_ds in ds_m.keys():
        to_ds = ds_m[from_ds]
        data_item, d = vecs_io.fvecs_read(os.path.join(basic_dir, from_ds, '%s_data_item.fvecs' % from_ds))
        query_item, d = vecs_io.fvecs_read(os.path.join(basic_dir, from_ds, '%s_query_item.fvecs' % from_ds))
        print("len ", data_item.shape)
        print(data_item.shape)

        queryID = get_sample_queryID(
            '{}-n_sample_item_5000-sample_topk_200'.format(from_ds))
        single_query_item = data_item[int(queryID), :]
        print(query_item.shape)
        query_item[replace_queryID] = single_query_item
        # query_item = query_item[[0]]

        vecs_io.fvecs_write(os.path.join(basic_dir, to_ds, '%s_query_item_convert.fvecs' % to_ds), query_item)
