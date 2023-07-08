import numpy as np
import os
import vecs_io


def delete_file_if_exist(dire):
    if os.path.exists(dire):
        command = 'rm -rf %s' % dire
        print(command)
        os.system(command)


if __name__ == '__main__':
    ds_m = {'amazon-home-kitchen-150d': 'amazon-home-kitchen-150d-update-user',
            'yelp_weighted_query-150d': 'yelp-150d-update-user'}
    n_update_user_m = {'amazon-home-kitchen-150d': 49248,
                       'yelp_weighted_query-150d': 42931}
    n_remain_user_m = {'amazon-home-kitchen-150d': 2462362,
                       'yelp_weighted_query-150d': 2146526}
    # basic_dir = '/home/bianzheng/Dataset/ReverseMIPS'
    basic_dir = '/home/zhengbian/Dataset/ReverseMIPS'

    for from_ds in ds_m.keys():
        to_ds = ds_m[from_ds]
        data_item, d = vecs_io.fvecs_read(os.path.join(basic_dir, from_ds, '%s_data_item.fvecs' % from_ds))
        print("data item ", data_item.shape)

        query_item, d = vecs_io.fvecs_read(os.path.join(basic_dir, from_ds, '%s_query_item.fvecs' % from_ds))
        print("query item ", query_item.shape)
        # query_item = query_item[[0]]

        old_user, d = vecs_io.fvecs_read(os.path.join(basic_dir, from_ds, '%s_user.fvecs' % from_ds))
        print("old user", old_user.shape)

        n_update_user = n_update_user_m[from_ds]
        remain_userID_l = np.arange(n_update_user, old_user.shape[0], 1)
        assert len(remain_userID_l) == n_remain_user_m[from_ds]
        user = old_user[remain_userID_l]

        update_userID_l = np.arange(0, n_update_user, 1)
        update_user = old_user[update_userID_l]

        delete_file_if_exist(os.path.join(basic_dir, to_ds))
        os.mkdir(os.path.join(basic_dir, to_ds))
        vecs_io.fvecs_write(os.path.join(basic_dir, to_ds, '%s_data_item.fvecs' % to_ds), data_item)
        vecs_io.fvecs_write(os.path.join(basic_dir, to_ds, '%s_user_update.fvecs' % to_ds), update_user)
        vecs_io.fvecs_write(os.path.join(basic_dir, to_ds, '%s_query_item.fvecs' % to_ds), query_item)
        vecs_io.fvecs_write(os.path.join(basic_dir, to_ds, '%s_user.fvecs' % to_ds), user)
