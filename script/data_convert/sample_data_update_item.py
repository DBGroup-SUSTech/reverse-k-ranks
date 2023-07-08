import numpy as np
import os
import vecs_io


def delete_file_if_exist(dire):
    if os.path.exists(dire):
        command = 'rm -rf %s' % dire
        print(command)
        os.system(command)


if __name__ == '__main__':
    ds_m = {'amazon-home-kitchen-150d': 'amazon-home-kitchen-150d-update-item',
            'yelp_weighted_query-150d': 'yelp-150d-update-item'}
    n_update_item_m = {'amazon-home-kitchen-150d': 8025,
                       'yelp_weighted_query-150d': 3130}
    n_remain_item_m = {'amazon-home-kitchen-150d': 401218,
                       'yelp_weighted_query-150d': 156455}
    # basic_dir = '/home/bianzheng/Dataset/ReverseMIPS'
    basic_dir = '/home/zhengbian/Dataset/ReverseMIPS'

    for from_ds in ds_m.keys():
        to_ds = ds_m[from_ds]
        old_data_item, d = vecs_io.fvecs_read(os.path.join(basic_dir, from_ds, '%s_data_item.fvecs' % from_ds))
        print("old data item ", old_data_item.shape)

        n_update_item = n_update_item_m[from_ds]

        data_itemID_l = np.arange(n_update_item, old_data_item.shape[0], 1)
        assert len(data_itemID_l) == n_remain_item_m[from_ds]
        data_item = old_data_item[data_itemID_l]

        update_itemID_l = np.arange(0, n_update_item, 1)
        update_item = old_data_item[update_itemID_l]

        query_item, d = vecs_io.fvecs_read(os.path.join(basic_dir, from_ds, '%s_query_item.fvecs' % from_ds))
        print("query item ", query_item.shape)
        # query_item = query_item[[0]]

        user, d = vecs_io.fvecs_read(os.path.join(basic_dir, from_ds, '%s_user.fvecs' % from_ds))
        print("user", user.shape)

        delete_file_if_exist(os.path.join(basic_dir, to_ds))
        os.mkdir(os.path.join(basic_dir, to_ds))
        vecs_io.fvecs_write(os.path.join(basic_dir, to_ds, '%s_data_item.fvecs' % to_ds), data_item)
        vecs_io.fvecs_write(os.path.join(basic_dir, to_ds, '%s_data_item_update.fvecs' % to_ds), update_item)
        vecs_io.fvecs_write(os.path.join(basic_dir, to_ds, '%s_query_item.fvecs' % to_ds), query_item)
        vecs_io.fvecs_write(os.path.join(basic_dir, to_ds, '%s_user.fvecs' % to_ds), user)
